#include "features.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>

#include "config.hpp"

namespace era {
namespace {

// pandas skips NaN in .mean(); an all-NaN group yields NaN, which is what lets
// the caller fall through to the next .fillna() in the chain.
double nanmean(const std::vector<double>& values) {
    double total = 0.0;
    std::size_t count = 0;
    for (const double v : values) {
        if (!std::isnan(v)) {
            total += v;
            ++count;
        }
    }
    return count ? total / static_cast<double>(count) : std::nan("");
}

// Group mean broadcast back over rows, i.e. df.groupby(key)[col].transform("mean").
std::vector<double> group_mean_transform(const std::vector<std::string>& keys,
                                         const std::vector<double>& values) {
    std::map<std::string, std::pair<double, std::size_t>> sums;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isnan(values[i])) {
            auto& entry = sums[keys[i]];
            entry.first += values[i];
            entry.second += 1;
        }
    }
    std::vector<double> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        auto it = sums.find(keys[i]);
        out[i] = (it != sums.end() && it->second.second)
                     ? it->second.first / static_cast<double>(it->second.second)
                     : std::nan("");
    }
    return out;
}

const std::vector<double>& column(const Frame& df, const std::string& name) {
    auto it = df.num.find(name);
    if (it == df.num.end()) {
        throw std::runtime_error("missing column: " + name);
    }
    return it->second;
}

}  // namespace

std::string decade_of(const std::string& season_year) {
    const int start = std::stoi(season_year.substr(0, 4));
    return std::to_string(start / 10 * 10) + "s";
}

std::vector<double> compute_age(const Frame& df) {
    const std::size_t n = df.size();
    std::vector<double> age(n, std::nan(""));
    std::vector<std::string> decade(n);
    for (std::size_t i = 0; i < n; ++i) {
        decade[i] = decade_of(df.season_year[i]);
        const std::string& birth = df.birth_date[i];
        if (birth.size() >= 4 && std::all_of(birth.begin(), birth.begin() + 4,
                                             [](unsigned char c) { return std::isdigit(c); })) {
            age[i] = std::stoi(df.season_year[i].substr(0, 4)) - std::stoi(birth.substr(0, 4));
        }
    }

    const std::vector<double> dec_mean = group_mean_transform(decade, age);
    const double overall_mean = nanmean(age);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(age[i])) {
            age[i] = dec_mean[i];
        }
        if (std::isnan(age[i])) {
            age[i] = overall_mean;
        }
        if (std::isnan(age[i])) {
            age[i] = 0.0;
        }
    }
    return age;
}

std::vector<double> build_value_target(const Frame& df) {
    const std::size_t n = df.size();
    std::vector<double> raw(n, 0.0);
    for (const std::string& col : VALUE_ALWAYS) {
        const std::vector<double>& values = column(df, col);
        for (std::size_t i = 0; i < n; ++i) {
            raw[i] += std::isnan(values[i]) ? 0.0 : values[i];
        }
    }
    for (const auto& [col, weight] : VALUE_CONDITIONAL) {
        const std::vector<double>& values = column(df, col);
        for (std::size_t i = 0; i < n; ++i) {
            raw[i] += (std::isnan(values[i]) ? 0.0 : values[i]) * weight;
        }
    }

    // Per-season z-score with ddof=0; a zero-variance season maps to all zeros.
    std::map<std::string, std::vector<std::size_t>> by_season;
    for (std::size_t i = 0; i < n; ++i) {
        by_season[df.season_year[i]].push_back(i);
    }
    std::vector<double> out(n, 0.0);
    for (const auto& [season, rows] : by_season) {
        double mean = 0.0;
        for (const std::size_t i : rows) {
            mean += raw[i];
        }
        mean /= static_cast<double>(rows.size());
        double variance = 0.0;
        for (const std::size_t i : rows) {
            variance += (raw[i] - mean) * (raw[i] - mean);
        }
        const double std_dev = std::sqrt(variance / static_cast<double>(rows.size()));
        for (const std::size_t i : rows) {
            out[i] = std_dev > 0.0 ? (raw[i] - mean) / std_dev : 0.0;
        }
    }
    return out;
}

std::map<std::string, std::vector<double>> build_position_oh(const Frame& df) {
    std::map<std::string, std::vector<double>> out;
    for (const std::string& position : POSITIONS) {
        out["pos_" + position] = std::vector<double>(df.size(), 0.0);
    }
    for (std::size_t i = 0; i < df.size(); ++i) {
        const std::string& raw = df.position[i];
        const bool known = std::find(POSITIONS.begin(), POSITIONS.end(), raw) != POSITIONS.end();
        out["pos_" + (known && !raw.empty() ? raw : std::string("Unknown"))][i] = 1.0;
    }
    return out;
}

Frame engineer_features(const Frame& df) {
    Frame out = df;
    out.era.resize(out.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        out.era[i] = decade_of(out.season_year[i]);
    }
    out.num["age"] = compute_age(out);

    // Impute numeric features with decade mean (historical reality: stl/blk/3pt
    // not recorded early; advanced metrics absent pre-1970s). If an entire era
    // lacks a stat (fg3_pct pre-1979, stl/blk pre-1973, tov/net_rating/usg pre-1977),
    // the decade mean is NaN -> fall back to the global column mean.
    for (const std::string& col : NUMERIC_FEATURES) {
        std::vector<double>& values = out.num.at(col);
        const std::vector<double> dec_mean = group_mean_transform(out.era, values);
        const double global_mean = nanmean(values);
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (std::isnan(values[i])) {
                values[i] = dec_mean[i];
            }
            if (std::isnan(values[i])) {
                values[i] = global_mean;
            }
        }
    }

    for (auto& [name, values] : build_position_oh(out)) {
        out.num[name] = std::move(values);
    }
    out.num["value_target"] = build_value_target(out);
    return out;
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> stratified_split(
    const std::vector<int64_t>& indices, const std::vector<std::string>& strata, double test_frac,
    std::mt19937_64& rng) {
    std::map<std::string, std::vector<int64_t>> by_class;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        by_class[strata[i]].push_back(indices[i]);
    }
    std::vector<int64_t> keep;
    std::vector<int64_t> held_out;
    for (auto& [label, members] : by_class) {
        std::shuffle(members.begin(), members.end(), rng);
        auto n_out = static_cast<std::size_t>(
            std::llround(static_cast<double>(members.size()) * test_frac));
        if (members.size() >= 2) {
            n_out = std::clamp<std::size_t>(n_out, 1, members.size() - 1);
        }
        held_out.insert(held_out.end(), members.begin(), members.begin() + n_out);
        keep.insert(keep.end(), members.begin() + n_out, members.end());
    }
    std::sort(keep.begin(), keep.end());
    std::sort(held_out.begin(), held_out.end());
    return {keep, held_out};
}

Splits make_splits(const Frame& df) {
    std::mt19937_64 rng(SEED);
    std::vector<int64_t> all(df.era.size());
    std::iota(all.begin(), all.end(), 0);

    auto [train, test] = stratified_split(all, df.era, TEST_FRAC, rng);

    std::vector<std::string> train_era(train.size());
    for (std::size_t i = 0; i < train.size(); ++i) {
        train_era[i] = df.era[train[i]];
    }
    auto [final_train, val] = stratified_split(train, train_era, VAL_FRAC, rng);

    return Splits{final_train, val, test};
}

std::vector<std::string> feature_columns(const Frame& df) {
    std::vector<std::string> cols = NUMERIC_FEATURES;
    cols.push_back("age");
    for (const std::string& position : POSITIONS) {
        cols.push_back("pos_" + position);
    }
    std::vector<std::string> missing;
    for (const std::string& col : cols) {
        if (!df.num.count(col)) {
            missing.push_back(col);
        }
    }
    if (!missing.empty()) {
        std::string message = "Missing feature columns:";
        for (const std::string& col : missing) {
            message += " " + col;
        }
        throw std::runtime_error(message);
    }
    return cols;
}

Scaler fit_scaler(const Frame& df, const std::vector<int64_t>& idx_train,
                  const std::vector<std::string>& cols) {
    Scaler scaler;
    for (const std::string& col : cols) {
        const std::vector<double>& values = column(df, col);
        double mean = 0.0;
        for (const int64_t i : idx_train) {
            mean += values[i];
        }
        mean /= static_cast<double>(idx_train.size());
        double variance = 0.0;
        for (const int64_t i : idx_train) {
            variance += (values[i] - mean) * (values[i] - mean);
        }
        const double std_dev = std::sqrt(variance / static_cast<double>(idx_train.size()));
        scaler.mean.push_back(mean);
        // StandardScaler leaves constant columns untouched rather than dividing by zero.
        scaler.scale.push_back(std_dev > 0.0 ? std_dev : 1.0);
    }
    return scaler;
}

Matrix transform_features(const Frame& df, const Scaler& scaler,
                          const std::vector<std::string>& cols) {
    Matrix out;
    out.rows = static_cast<int64_t>(df.size());
    out.cols = static_cast<int64_t>(cols.size());
    out.data.resize(static_cast<std::size_t>(out.rows) * out.cols);
    for (int64_t c = 0; c < out.cols; ++c) {
        const std::vector<double>& values = column(df, cols[c]);
        for (int64_t r = 0; r < out.rows; ++r) {
            out.data[r * out.cols + c] =
                static_cast<float>((values[r] - scaler.mean[c]) / scaler.scale[c]);
        }
    }
    return out;
}

std::vector<int64_t> era_codes(const Frame& df) {
    std::vector<int64_t> codes(df.size());
    for (std::size_t i = 0; i < df.size(); ++i) {
        const auto it = std::find(ERA_ORDER.begin(), ERA_ORDER.end(), df.era[i]);
        if (it == ERA_ORDER.end()) {
            throw std::runtime_error("unknown era: " + df.era[i]);
        }
        codes[i] = std::distance(ERA_ORDER.begin(), it);
    }
    return codes;
}

}  // namespace era
