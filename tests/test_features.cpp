// Unit tests for feature engineering.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <set>

#include "config.hpp"
#include "features.hpp"
#include "frame.hpp"

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

TEST(Features, DecadeOf) {
    EXPECT_EQ(era::decade_of("1946-47"), "1940s");
    EXPECT_EQ(era::decade_of("2002-03"), "2000s");
    EXPECT_EQ(era::decade_of("2025-26"), "2020s");
}

TEST(Features, ComputeAgeKnownBirth) {
    era::Frame df;
    df.season_year = {"2002-03", "1990-91", "2002-03"};
    df.birth_date = {"1984-12-30", "1963-02-17", ""};

    const std::vector<double> ages = era::compute_age(df);
    EXPECT_DOUBLE_EQ(ages[0], 18.0);
    EXPECT_DOUBLE_EQ(ages[1], 27.0);
    EXPECT_NEAR(ages[2], ages[0], 1e-9);  // decade mean fallback
}

TEST(Features, BuildValueTargetPerSeasonZScore) {
    era::Frame df;
    df.season_year = {"2020-21", "2020-21", "2020-21", "2010-11", "2010-11", "2010-11"};
    df.num["avg_pts"] = {10.0, 20.0, 30.0, 15.0, 15.0, 15.0};
    df.num["avg_reb"] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    df.num["avg_ast"] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    df.num["avg_stl"] = std::vector<double>(6, kNaN);
    df.num["avg_blk"] = std::vector<double>(6, kNaN);
    df.num["avg_tov"] = std::vector<double>(6, kNaN);

    const std::vector<double> target = era::build_value_target(df);
    const double sd = std::sqrt((100.0 + 0.0 + 100.0) / 3.0);  // ddof=0
    EXPECT_NEAR(target[0], (10.0 - 20.0) / sd, 1e-6);
    EXPECT_NEAR(target[1], 0.0, 1e-6);
    EXPECT_NEAR(target[2], (30.0 - 20.0) / sd, 1e-6);
    for (std::size_t i = 3; i < 6; ++i) {
        EXPECT_NEAR(target[i], 0.0, 1e-6);
    }
}

TEST(Features, EngineerFeaturesNoNan) {
    era::Frame df;
    df.player_id = {1, 2, 3};
    df.season_year = {"1960-61", "2000-01", "2020-21"};
    df.birth_date = {"1940-01-01", "1980-01-01", "1998-01-01"};
    df.position = {"C", "F", "G"};
    df.num["gp"] = {70, 82, 75};
    df.num["avg_min"] = {30.0, 32.0, 34.0};
    df.num["avg_pts"] = {20.0, 22.0, 25.0};
    df.num["avg_reb"] = {10.0, 8.0, 7.0};
    df.num["avg_ast"] = {3.0, 5.0, 7.0};
    df.num["avg_stl"] = {kNaN, 1.5, 1.2};
    df.num["avg_blk"] = {kNaN, 0.5, 0.6};
    df.num["avg_tov"] = {kNaN, 2.5, 2.8};
    df.num["fg_pct"] = {0.45, 0.46, 0.48};
    df.num["fg3_pct"] = {kNaN, 0.35, 0.38};
    df.num["ft_pct"] = {0.75, 0.78, 0.80};
    df.num["avg_net_rating"] = {kNaN, 3.0, 4.0};
    df.num["avg_ts_pct"] = {0.50, 0.55, 0.58};
    df.num["avg_usg_pct"] = {kNaN, 25.0, 27.0};

    const era::Frame out = era::engineer_features(df);
    const std::vector<std::string> cols = era::feature_columns(out);
    for (const std::string& col : cols) {
        for (const double value : out.num.at(col)) {
            EXPECT_FALSE(std::isnan(value)) << "NaN left in " << col;
        }
    }

    const std::set<std::string> present(cols.begin(), cols.end());
    for (const std::string& expected : {"avg_min", "gp", "fg_pct", "age", "pos_G", "pos_F",
                                        "pos_C", "pos_Unknown"}) {
        EXPECT_TRUE(present.count(expected)) << "missing " << expected;
    }
}

TEST(Features, MakeSplitsStratifiedDisjoint) {
    era::Frame df;
    for (const std::string& label : {"1940s", "2000s", "2020s"}) {
        for (int i = 0; i < 50; ++i) {
            df.era.push_back(label);
        }
    }

    const era::Splits splits = era::make_splits(df);
    const std::set<int64_t> train(splits.train.begin(), splits.train.end());
    const std::set<int64_t> val(splits.val.begin(), splits.val.end());
    const std::set<int64_t> test(splits.test.begin(), splits.test.end());

    const auto disjoint = [](const std::set<int64_t>& a, const std::set<int64_t>& b) {
        return std::none_of(a.begin(), a.end(), [&b](int64_t i) { return b.count(i) > 0; });
    };
    EXPECT_TRUE(disjoint(train, val));
    EXPECT_TRUE(disjoint(train, test));
    EXPECT_TRUE(disjoint(val, test));
    EXPECT_EQ(train.size() + val.size() + test.size(), 150u);

    for (const std::vector<int64_t>* split : {&splits.train, &splits.val, &splits.test}) {
        std::set<std::string> eras;
        for (const int64_t i : *split) {
            eras.insert(df.era[i]);
        }
        EXPECT_EQ(eras, (std::set<std::string>{"1940s", "2000s", "2020s"}));
    }
}

TEST(Features, BuildPositionOneHot) {
    era::Frame df;
    df.position = {"G", "F", "C", "", "PG"};

    const auto oh = era::build_position_oh(df);
    EXPECT_EQ(oh.size(), 4u);
    EXPECT_DOUBLE_EQ(oh.at("pos_G")[0], 1.0);
    EXPECT_DOUBLE_EQ(oh.at("pos_F")[0], 0.0);
    EXPECT_DOUBLE_EQ(oh.at("pos_Unknown")[3], 1.0);  // NULL -> Unknown
    EXPECT_DOUBLE_EQ(oh.at("pos_Unknown")[4], 1.0);  // unknown position -> Unknown
}
