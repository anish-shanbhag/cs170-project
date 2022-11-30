#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <time.h>
#include <string>

using namespace std;

string type = "medium";
const int nodes = 300;
const int input_offset = 260;
const int steps = 500000000;
const int portion = 0;
const int inputs_per_portion = 35;

int get_k_from_output(int num) {
    ifstream xfp("cpp-outputs/" + type + to_string(num) + ".out");
    int k = 0;
    for (int i = 0; i < nodes; i++) {
        int val;
        xfp >> val;
        if (val > k) {
            k = val;
        }
    }
    xfp.close();
    return k;
}

void get_initial_state_from_output(
    int num,
    int k_max,
    int w[nodes][nodes],
    int x[nodes],
    int p[],
    unordered_set<int> s[],
    double b[],
    double* b_sum,
    double* d,
    double* score
) {
    *score = 0;
    *b_sum = 0;
    *d = 0;

    ifstream fp("weights/" + type + to_string(num) + ".txt");
    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            fp >> w[i][j];
        }
    }

    ifstream xfp("cpp-outputs/" + type + to_string(num) + ".out");
    for (int i = 0; i < nodes; i++) {
        xfp >> x[i];
        x[i] = min(x[i], k_max);
        p[x[i] - 1]++;
        s[x[i] - 1].insert(i);
    }
    xfp.close();

    for (int i = 0; i < nodes; i++) {
        for (int j = i + 1; j < nodes; j++) {
            if (x[i] == x[j]) {
                *score += w[i][j];
            }
        }
    }

    *score += 100 * exp(0.5 * k_max);

    for (int i = 0; i < k_max; i++) {
        b[i] = (double) p[i] / nodes - 1.0 / k_max;
    }

    for (int i = 0; i < k_max; i++) {
        *b_sum += b[i] * b[i];
    }

    *d = exp(70 * sqrt(*b_sum));
    *score += *d;
}

void anneal(int num, int k_max) {
    double T_min = 4.5;
    double T_max = 33000;
    int w[nodes][nodes] = {};
    int x[nodes] = {};
    int p[k_max] = {};
    unordered_set<int> s[k_max];
    double b[k_max] = {};
    double b_sum = 0;
    double d = 0;
    double score = 0;

    get_initial_state_from_output(
        num,
        k_max,
        w,
        x,
        p,
        s,
        b,
        &b_sum,
        &d,
        &score
    );

    // simulated annealing
    clock_t start_time = clock();
    double best_score = 1000000000.0;
    double best_x[nodes];
    double T = T_max;
    double T_factor = -log(T_max / T_min);
    random_device dev;
    mt19937 rng(dev());
    uniform_int_distribution<mt19937::result_type> x_dist(0, nodes - 1);
    uniform_int_distribution<mt19937::result_type> k_dist(1, k_max);
    uniform_real_distribution<double> T_dist(0.0, 1.0);

    for (double step = 0; step < steps; step++) {
        T = T_max * exp(T_factor * step / steps);
        int i = x_dist(rng);
        double delta = 0;

        for (int j : s[x[i] - 1]) {
            delta -= w[i][j];
        }

        int new_x = k_dist(rng);
        while (new_x == x[i]) {
            new_x = k_dist(rng);
        }

        for (int j : s[new_x - 1]) {
            delta += w[i][j];
        }

        double new_b_sum = b_sum - pow(b[x[i] - 1], 2) - pow(b[new_x - 1], 2) + pow(b[x[i] - 1] - 1.0 / nodes, 2) + pow(b[new_x - 1] + 1.0 / nodes, 2);
        double new_d = exp(70 * sqrt(new_b_sum));
        delta += new_d - d;

        if (delta <= 0 || exp(-delta / T) > T_dist(rng)) {
            score += delta;
            b_sum = new_b_sum;
            d = new_d;
            p[x[i] - 1]--;
            p[new_x - 1]++;
            b[x[i] - 1] -= 1.0 / nodes;
            b[new_x - 1] += 1.0 / nodes;
            s[x[i] - 1].erase(i);
            s[new_x - 1].insert(i);
            x[i] = new_x;

            if (score < best_score) {
                best_score = score;
                for (int i = 0; i < nodes; i++) {
                    best_x[i] = x[i];
                }
                // cout << "step " << step << " score " << score << endl;
            }
        }
    }

    int k = get_k_from_output(num);
    int old_p[k] = {};
    unordered_set<int> old_s[k];
    double old_b[k] = {};
    double old_score = 0;
    get_initial_state_from_output(
        num,
        k,
        w,
        x,
        old_p,
        old_s,
        old_b,
        &b_sum,
        &d,
        &old_score
    );
    if (best_score < old_score) {
        cout << "NEW BEST SCORE (down from " << old_score << "): ";
        ofstream out("cpp-outputs/" + type + to_string(num) + ".out");
        for (int i = 0; i < nodes; i++) {
            out << best_x[i] << endl;
        }
        out.close();
    } else {
        cout << "Skipping score ";
    }
    cout << best_score << " for input " << type << num << " with k_max = " << k_max << " (" << (clock() - start_time) / CLOCKS_PER_SEC << " sec)" << endl;
}

void anneal(int num, double best_scores[]) {
    double score_to_beat = best_scores[input_offset + num - 1];
    int k_actual_max = max(2, floor(2 * log(score_to_beat / 100.0)));
    int k_min = max(2, k_actual_max - 5);
    for (int k_max = k_min; k_max <= k_actual_max; k_max++) {
        anneal(num, k_max);
    }
}

int main() {
    ifstream sfp("scores.txt");
    double best_scores[260 * 3];
    for (int i = 0; i < 260 * 3; i++) {
        sfp >> best_scores[i];
    }
    sfp.close();

    for (int i = 1 + portion * inputs_per_portion; i < 1 + (portion + 1) * inputs_per_portion; i++) {
        anneal(i, best_scores);
    }
    return 0;
}

// g++ -o rideThatSlay anneal.cpp -O3
