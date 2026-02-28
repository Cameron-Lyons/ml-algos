#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace {

double sigmoidSafe(double v) {
  const double clamped = std::clamp(v, -60.0, 60.0);
  return 1.0 / (1.0 + std::exp(-clamped));
}

double dotProduct(const Vector &a, const Vector &b) {
  double out = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    out += a[i] * b[i];
  }
  return out;
}

class TrainableAdapterHead {
private:
  Vector adapterScale_;
  Vector adapterShift_;
  Vector weights_;
  double bias_ = 0.0;
  bool initialized_ = false;

  void ensureInitialized(size_t dim) {
    if (initialized_ && weights_.size() == dim) {
      return;
    }
    adapterScale_.assign(dim, 1.0);
    adapterShift_.assign(dim, 0.0);
    weights_.assign(dim, 0.0);
    bias_ = 0.0;
    initialized_ = true;
  }

  Vector adapt(const Vector &raw) const {
    Vector adapted(raw.size(), 0.0);
    for (size_t j = 0; j < raw.size(); j++) {
      adapted[j] = (adapterScale_[j] * raw[j]) + adapterShift_[j];
    }
    return adapted;
  }

  double linear(const Vector &raw) const {
    Vector adapted = adapt(raw);
    return dotProduct(weights_, adapted) + bias_;
  }

public:
  void fit(const Matrix &rawFeatures, const Vector &y, bool classification,
           int epochs = 260, double learningRate = 0.015,
           double l2 = 5e-4) {
    if (rawFeatures.empty() || y.empty()) {
      initialized_ = false;
      adapterScale_.clear();
      adapterShift_.clear();
      weights_.clear();
      bias_ = 0.0;
      return;
    }

    const size_t dim = rawFeatures.front().size();
    ensureInitialized(dim);

    std::vector<size_t> order(rawFeatures.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::mt19937 rng(42);

    for (int epoch = 0; epoch < epochs; epoch++) {
      std::shuffle(order.begin(), order.end(), rng);
      const double lr =
          learningRate * (1.0 - (0.75 * static_cast<double>(epoch) /
                                   static_cast<double>(std::max(1, epochs))));

      for (size_t idx : order) {
        const Vector &raw = rawFeatures[idx];
        Vector adapted = adapt(raw);
        const double linearScore = dotProduct(weights_, adapted) + bias_;

        double grad = 0.0;
        if (classification) {
          grad = sigmoidSafe(linearScore) - y[idx];
        } else {
          grad = linearScore - y[idx];
        }
        grad = std::clamp(grad, -10.0, 10.0);

        for (size_t j = 0; j < dim; j++) {
          const double wOld = weights_[j];
          const double gradW = (grad * adapted[j]) + (l2 * weights_[j]);
          const double gradScale =
              (grad * wOld * raw[j]) + (l2 * (adapterScale_[j] - 1.0));
          const double gradShift = (grad * wOld) + (l2 * adapterShift_[j]);

          weights_[j] -= lr * gradW;
          adapterScale_[j] -= lr * gradScale;
          adapterShift_[j] -= lr * gradShift;
        }
        bias_ -= lr * grad;
      }
    }
  }

  double predictRegression(const Vector &raw) const {
    return linear(raw);
  }

  double predictClassification(const Vector &raw) const {
    const double p = sigmoidSafe(linear(raw));
    return p >= 0.5 ? 1.0 : 0.0;
  }

  const Vector &getWeights() const { return weights_; }
  double getBias() const { return bias_; }
  const Vector &getAdapterScale() const { return adapterScale_; }
  const Vector &getAdapterShift() const { return adapterShift_; }

  void setState(const Vector &weights, double bias, const Vector &adapterScale,
                const Vector &adapterShift) {
    weights_ = weights;
    bias_ = bias;
    adapterScale_ = adapterScale;
    adapterShift_ = adapterShift;
    initialized_ = !weights_.empty() && (weights_.size() == adapterScale_.size()) &&
                   (weights_.size() == adapterShift_.size());
  }
};

class TinyCNNEncoder {
private:
  int nFilters_;
  int kernelSize_;
  Matrix filters_;
  Vector bias_;
  bool initialized_ = false;

public:
  TinyCNNEncoder(int nFilters = 8, int kernelSize = 3)
      : nFilters_(std::max(1, nFilters)), kernelSize_(std::max(1, kernelSize)) {}

  void ensureInitialized(size_t inputSize) {
    if (initialized_) {
      return;
    }

    const int effectiveKernel =
        std::max(1, std::min(kernelSize_, static_cast<int>(inputSize)));

    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.15);

    filters_.assign(static_cast<size_t>(nFilters_),
                    Vector(static_cast<size_t>(effectiveKernel), 0.0));
    bias_.assign(static_cast<size_t>(nFilters_), 0.0);

    for (auto &filter : filters_) {
      for (double &w : filter) {
        w = dist(rng);
      }
    }
    for (double &b : bias_) {
      b = dist(rng);
    }
    initialized_ = true;
  }

  Vector encode(const Vector &x) {
    ensureInitialized(x.size());

    Vector features(static_cast<size_t>(nFilters_), 0.0);
    const size_t kernel = filters_.front().size();
    const size_t steps = x.size() >= kernel ? (x.size() - kernel + 1) : 1;

    for (int f = 0; f < nFilters_; f++) {
      double pooled = 0.0;
      for (size_t start = 0; start < steps; start++) {
        double conv = bias_[static_cast<size_t>(f)];
        for (size_t k = 0; k < kernel; k++) {
          const size_t idx = std::min(start + k, x.size() - 1);
          conv += x[idx] * filters_[static_cast<size_t>(f)][k];
        }
        pooled += std::max(0.0, conv);
      }
      features[static_cast<size_t>(f)] = pooled / static_cast<double>(steps);
    }

    return features;
  }

  int nFilters() const { return nFilters_; }
  int kernelSize() const { return kernelSize_; }
};

class TinyRNNEncoder {
private:
  int hidden_;
  Matrix wH_;
  Vector wX_;
  Vector b_;

public:
  explicit TinyRNNEncoder(int hidden = 12)
      : hidden_(std::max(2, hidden)),
        wH_(static_cast<size_t>(hidden_), Vector(static_cast<size_t>(hidden_), 0.0)),
        wX_(static_cast<size_t>(hidden_), 0.0),
        b_(static_cast<size_t>(hidden_), 0.0) {
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 0.2);
    for (double &v : wX_) {
      v = dist(rng);
    }
    for (double &v : b_) {
      v = dist(rng);
    }
    for (auto &row : wH_) {
      for (double &v : row) {
        v = dist(rng) / static_cast<double>(hidden_);
      }
    }
  }

  Vector encode(const Vector &x) const {
    Vector h(static_cast<size_t>(hidden_), 0.0);
    Vector hNext(static_cast<size_t>(hidden_), 0.0);

    for (double value : x) {
      for (int j = 0; j < hidden_; j++) {
        double pre = b_[static_cast<size_t>(j)] +
                     (wX_[static_cast<size_t>(j)] * value);
        for (int k = 0; k < hidden_; k++) {
          pre += wH_[static_cast<size_t>(j)][static_cast<size_t>(k)] *
                 h[static_cast<size_t>(k)];
        }
        hNext[static_cast<size_t>(j)] = std::tanh(pre);
      }
      h.swap(hNext);
    }

    return h;
  }

  int hidden() const { return hidden_; }
};

class TinyLSTMEncoder {
private:
  int hidden_;
  Vector wiX_;
  Vector wfX_;
  Vector woX_;
  Vector wgX_;
  Matrix wiH_;
  Matrix wfH_;
  Matrix woH_;
  Matrix wgH_;
  Vector bi_;
  Vector bf_;
  Vector bo_;
  Vector bg_;

public:
  explicit TinyLSTMEncoder(int hidden = 10)
      : hidden_(std::max(2, hidden)), wiX_(static_cast<size_t>(hidden_), 0.0),
        wfX_(static_cast<size_t>(hidden_), 0.0),
        woX_(static_cast<size_t>(hidden_), 0.0),
        wgX_(static_cast<size_t>(hidden_), 0.0),
        wiH_(static_cast<size_t>(hidden_), Vector(static_cast<size_t>(hidden_), 0.0)),
        wfH_(static_cast<size_t>(hidden_), Vector(static_cast<size_t>(hidden_), 0.0)),
        woH_(static_cast<size_t>(hidden_), Vector(static_cast<size_t>(hidden_), 0.0)),
        wgH_(static_cast<size_t>(hidden_), Vector(static_cast<size_t>(hidden_), 0.0)),
        bi_(static_cast<size_t>(hidden_), 0.0),
        bf_(static_cast<size_t>(hidden_), 0.0),
        bo_(static_cast<size_t>(hidden_), 0.0),
        bg_(static_cast<size_t>(hidden_), 0.0) {
    std::mt19937 rng(321);
    std::normal_distribution<double> dist(0.0, 0.18);

    auto initVec = [&](Vector &v) {
      for (double &x : v) {
        x = dist(rng);
      }
    };
    auto initMat = [&](Matrix &m) {
      for (auto &row : m) {
        for (double &x : row) {
          x = dist(rng) / static_cast<double>(hidden_);
        }
      }
    };

    initVec(wiX_);
    initVec(wfX_);
    initVec(woX_);
    initVec(wgX_);
    initVec(bi_);
    initVec(bf_);
    initVec(bo_);
    initVec(bg_);
    initMat(wiH_);
    initMat(wfH_);
    initMat(woH_);
    initMat(wgH_);
  }

  Vector encode(const Vector &x) const {
    Vector h(static_cast<size_t>(hidden_), 0.0);
    Vector c(static_cast<size_t>(hidden_), 0.0);

    for (double value : x) {
      for (int j = 0; j < hidden_; j++) {
        double preI = bi_[static_cast<size_t>(j)] +
                      (wiX_[static_cast<size_t>(j)] * value);
        double preF = bf_[static_cast<size_t>(j)] +
                      (wfX_[static_cast<size_t>(j)] * value);
        double preO = bo_[static_cast<size_t>(j)] +
                      (woX_[static_cast<size_t>(j)] * value);
        double preG = bg_[static_cast<size_t>(j)] +
                      (wgX_[static_cast<size_t>(j)] * value);

        for (int k = 0; k < hidden_; k++) {
          const double hk = h[static_cast<size_t>(k)];
          preI += wiH_[static_cast<size_t>(j)][static_cast<size_t>(k)] * hk;
          preF += wfH_[static_cast<size_t>(j)][static_cast<size_t>(k)] * hk;
          preO += woH_[static_cast<size_t>(j)][static_cast<size_t>(k)] * hk;
          preG += wgH_[static_cast<size_t>(j)][static_cast<size_t>(k)] * hk;
        }

        const double i = sigmoidSafe(preI);
        const double f = sigmoidSafe(preF);
        const double o = sigmoidSafe(preO);
        const double g = std::tanh(preG);

        c[static_cast<size_t>(j)] = (f * c[static_cast<size_t>(j)]) + (i * g);
        h[static_cast<size_t>(j)] = o * std::tanh(c[static_cast<size_t>(j)]);
      }
    }

    return h;
  }

  int hidden() const { return hidden_; }
};

class TinyTransformerEncoder {
private:
  int hidden_;
  Vector qProj_;
  Vector kProj_;
  Vector phase_;
  Vector freq_;

public:
  explicit TinyTransformerEncoder(int hidden = 16)
      : hidden_(std::max(4, hidden)), qProj_(static_cast<size_t>(hidden_), 0.0),
        kProj_(static_cast<size_t>(hidden_), 0.0),
        phase_(static_cast<size_t>(hidden_), 0.0),
        freq_(static_cast<size_t>(hidden_), 0.0) {
    std::mt19937 rng(777);
    std::uniform_real_distribution<double> pDist(0.0, 2.0 * std::numbers::pi);
    std::uniform_real_distribution<double> fDist(0.1, 2.0);
    std::normal_distribution<double> wDist(0.0, 0.2);

    for (int i = 0; i < hidden_; i++) {
      qProj_[static_cast<size_t>(i)] = wDist(rng);
      kProj_[static_cast<size_t>(i)] = wDist(rng);
      phase_[static_cast<size_t>(i)] = pDist(rng);
      freq_[static_cast<size_t>(i)] = fDist(rng);
    }
  }

  Vector embedToken(double x) const {
    Vector e(static_cast<size_t>(hidden_), 0.0);
    for (int i = 0; i < hidden_; i++) {
      e[static_cast<size_t>(i)] =
          std::sin((freq_[static_cast<size_t>(i)] * x) +
                   phase_[static_cast<size_t>(i)]);
    }
    return e;
  }

  Vector encode(const Vector &x) const {
    if (x.empty()) {
      return Vector(static_cast<size_t>(hidden_), 0.0);
    }

    const size_t n = x.size();
    std::vector<Vector> tokens;
    tokens.reserve(n);
    for (double value : x) {
      tokens.push_back(embedToken(value));
    }

    Vector q(n, 0.0);
    Vector k(n, 0.0);
    for (size_t t = 0; t < n; t++) {
      q[t] = dotProduct(tokens[t], qProj_);
      k[t] = dotProduct(tokens[t], kProj_);
    }

    const double scale = 1.0 / std::sqrt(static_cast<double>(hidden_));
    std::vector<Vector> contexts(n, Vector(static_cast<size_t>(hidden_), 0.0));

    for (size_t i = 0; i < n; i++) {
      Vector scores(n, 0.0);
      double maxScore = -std::numeric_limits<double>::infinity();
      for (size_t j = 0; j < n; j++) {
        scores[j] = (q[i] * k[j]) * scale;
        maxScore = std::max(maxScore, scores[j]);
      }

      double denom = 0.0;
      for (size_t j = 0; j < n; j++) {
        scores[j] = std::exp(scores[j] - maxScore);
        denom += scores[j];
      }
      if (denom <= 0.0) {
        continue;
      }

      for (size_t j = 0; j < n; j++) {
        const double attn = scores[j] / denom;
        for (int d = 0; d < hidden_; d++) {
          contexts[i][static_cast<size_t>(d)] +=
              attn * tokens[j][static_cast<size_t>(d)];
        }
      }
    }

    Vector pooled(static_cast<size_t>(hidden_), 0.0);
    for (size_t i = 0; i < n; i++) {
      for (int d = 0; d < hidden_; d++) {
        pooled[static_cast<size_t>(d)] +=
            contexts[i][static_cast<size_t>(d)];
      }
    }

    const double invN = 1.0 / static_cast<double>(n);
    for (double &v : pooled) {
      v *= invN;
    }
    return pooled;
  }

  int hidden() const { return hidden_; }
};

template <typename Encoder>
Matrix encodeMatrix(Encoder &encoder, const Matrix &X) {
  Matrix H;
  H.reserve(X.size());
  for (const auto &x : X) {
    H.push_back(encoder.encode(x));
  }
  return H;
}

template <typename Encoder>
Vector encodeVector(Encoder &encoder, const Vector &x) {
  return encoder.encode(x);
}

} // namespace

class CNNRegressor {
private:
  TinyCNNEncoder encoder_;
  TrainableAdapterHead head_;

public:
  CNNRegressor(int filters = 8, int kernelSize = 3)
      : encoder_(filters, kernelSize), head_() {}

  void fit(const Matrix &X, const Vector &y) {
    Matrix H = encodeMatrix(encoder_, X);
    head_.fit(H, y, false, 280, 0.012);
  }

  double predict(const Vector &x) {
    return head_.predictRegression(encodeVector(encoder_, x));
  }

  const Vector &getHeadWeights() const { return head_.getWeights(); }
  double getHeadBias() const { return head_.getBias(); }
  const Vector &getAdapterScale() const { return head_.getAdapterScale(); }
  const Vector &getAdapterShift() const { return head_.getAdapterShift(); }
  int getFilters() const { return encoder_.nFilters(); }
  int getKernelSize() const { return encoder_.kernelSize(); }

  void setReadoutState(const Vector &weights, double bias,
                       const Vector &adapterScale,
                       const Vector &adapterShift) {
    head_.setState(weights, bias, adapterScale, adapterShift);
  }
};

class CNNClassifier {
private:
  TinyCNNEncoder encoder_;
  TrainableAdapterHead head_;

public:
  CNNClassifier(int filters = 8, int kernelSize = 3)
      : encoder_(filters, kernelSize), head_() {}

  void fit(const Matrix &X, const Vector &y) {
    Matrix H = encodeMatrix(encoder_, X);
    head_.fit(H, y, true, 320, 0.01);
  }

  double predict(const Vector &x) {
    return head_.predictClassification(encodeVector(encoder_, x));
  }

  const Vector &getHeadWeights() const { return head_.getWeights(); }
  double getHeadBias() const { return head_.getBias(); }
  const Vector &getAdapterScale() const { return head_.getAdapterScale(); }
  const Vector &getAdapterShift() const { return head_.getAdapterShift(); }
  int getFilters() const { return encoder_.nFilters(); }
  int getKernelSize() const { return encoder_.kernelSize(); }

  void setReadoutState(const Vector &weights, double bias,
                       const Vector &adapterScale,
                       const Vector &adapterShift) {
    head_.setState(weights, bias, adapterScale, adapterShift);
  }
};

class RNNRegressor {
private:
  TinyRNNEncoder encoder_;
  TrainableAdapterHead head_;

public:
  explicit RNNRegressor(int hidden = 12) : encoder_(hidden), head_() {}

  void fit(const Matrix &X, const Vector &y) {
    Matrix H = encodeMatrix(encoder_, X);
    head_.fit(H, y, false, 260, 0.01);
  }

  double predict(const Vector &x) {
    return head_.predictRegression(encodeVector(encoder_, x));
  }

  const Vector &getHeadWeights() const { return head_.getWeights(); }
  double getHeadBias() const { return head_.getBias(); }
  const Vector &getAdapterScale() const { return head_.getAdapterScale(); }
  const Vector &getAdapterShift() const { return head_.getAdapterShift(); }
  int getHidden() const { return encoder_.hidden(); }

  void setReadoutState(const Vector &weights, double bias,
                       const Vector &adapterScale,
                       const Vector &adapterShift) {
    head_.setState(weights, bias, adapterScale, adapterShift);
  }
};

class RNNClassifier {
private:
  TinyRNNEncoder encoder_;
  TrainableAdapterHead head_;

public:
  explicit RNNClassifier(int hidden = 12) : encoder_(hidden), head_() {}

  void fit(const Matrix &X, const Vector &y) {
    Matrix H = encodeMatrix(encoder_, X);
    head_.fit(H, y, true, 300, 0.009);
  }

  double predict(const Vector &x) {
    return head_.predictClassification(encodeVector(encoder_, x));
  }

  const Vector &getHeadWeights() const { return head_.getWeights(); }
  double getHeadBias() const { return head_.getBias(); }
  const Vector &getAdapterScale() const { return head_.getAdapterScale(); }
  const Vector &getAdapterShift() const { return head_.getAdapterShift(); }
  int getHidden() const { return encoder_.hidden(); }

  void setReadoutState(const Vector &weights, double bias,
                       const Vector &adapterScale,
                       const Vector &adapterShift) {
    head_.setState(weights, bias, adapterScale, adapterShift);
  }
};

class LSTMRegressor {
private:
  TinyLSTMEncoder encoder_;
  TrainableAdapterHead head_;

public:
  explicit LSTMRegressor(int hidden = 10) : encoder_(hidden), head_() {}

  void fit(const Matrix &X, const Vector &y) {
    Matrix H = encodeMatrix(encoder_, X);
    head_.fit(H, y, false, 260, 0.01);
  }

  double predict(const Vector &x) {
    return head_.predictRegression(encodeVector(encoder_, x));
  }

  const Vector &getHeadWeights() const { return head_.getWeights(); }
  double getHeadBias() const { return head_.getBias(); }
  const Vector &getAdapterScale() const { return head_.getAdapterScale(); }
  const Vector &getAdapterShift() const { return head_.getAdapterShift(); }
  int getHidden() const { return encoder_.hidden(); }

  void setReadoutState(const Vector &weights, double bias,
                       const Vector &adapterScale,
                       const Vector &adapterShift) {
    head_.setState(weights, bias, adapterScale, adapterShift);
  }
};

class LSTMClassifier {
private:
  TinyLSTMEncoder encoder_;
  TrainableAdapterHead head_;

public:
  explicit LSTMClassifier(int hidden = 10) : encoder_(hidden), head_() {}

  void fit(const Matrix &X, const Vector &y) {
    Matrix H = encodeMatrix(encoder_, X);
    head_.fit(H, y, true, 300, 0.009);
  }

  double predict(const Vector &x) {
    return head_.predictClassification(encodeVector(encoder_, x));
  }

  const Vector &getHeadWeights() const { return head_.getWeights(); }
  double getHeadBias() const { return head_.getBias(); }
  const Vector &getAdapterScale() const { return head_.getAdapterScale(); }
  const Vector &getAdapterShift() const { return head_.getAdapterShift(); }
  int getHidden() const { return encoder_.hidden(); }

  void setReadoutState(const Vector &weights, double bias,
                       const Vector &adapterScale,
                       const Vector &adapterShift) {
    head_.setState(weights, bias, adapterScale, adapterShift);
  }
};

class TransformerRegressor {
private:
  TinyTransformerEncoder encoder_;
  TrainableAdapterHead head_;

public:
  explicit TransformerRegressor(int hidden = 16)
      : encoder_(hidden), head_() {}

  void fit(const Matrix &X, const Vector &y) {
    Matrix H = encodeMatrix(encoder_, X);
    head_.fit(H, y, false, 260, 0.012);
  }

  double predict(const Vector &x) {
    return head_.predictRegression(encodeVector(encoder_, x));
  }

  const Vector &getHeadWeights() const { return head_.getWeights(); }
  double getHeadBias() const { return head_.getBias(); }
  const Vector &getAdapterScale() const { return head_.getAdapterScale(); }
  const Vector &getAdapterShift() const { return head_.getAdapterShift(); }
  int getHidden() const { return encoder_.hidden(); }

  void setReadoutState(const Vector &weights, double bias,
                       const Vector &adapterScale,
                       const Vector &adapterShift) {
    head_.setState(weights, bias, adapterScale, adapterShift);
  }
};

class TransformerClassifier {
private:
  TinyTransformerEncoder encoder_;
  TrainableAdapterHead head_;

public:
  explicit TransformerClassifier(int hidden = 16)
      : encoder_(hidden), head_() {}

  void fit(const Matrix &X, const Vector &y) {
    Matrix H = encodeMatrix(encoder_, X);
    head_.fit(H, y, true, 300, 0.01);
  }

  double predict(const Vector &x) {
    return head_.predictClassification(encodeVector(encoder_, x));
  }

  const Vector &getHeadWeights() const { return head_.getWeights(); }
  double getHeadBias() const { return head_.getBias(); }
  const Vector &getAdapterScale() const { return head_.getAdapterScale(); }
  const Vector &getAdapterShift() const { return head_.getAdapterShift(); }
  int getHidden() const { return encoder_.hidden(); }

  void setReadoutState(const Vector &weights, double bias,
                       const Vector &adapterScale,
                       const Vector &adapterShift) {
    head_.setState(weights, bias, adapterScale, adapterShift);
  }
};
