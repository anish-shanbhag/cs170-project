#include <iostream>
#include <fstream>
#include <unordered_set>
#include <random>
#include <time.h>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

using namespace std;

string type = "medium";
const int nodes = 300;
const int input_offset = 260;
const int steps = 500000000;
const bool run_all = false;
const bool try_to_break_ties = false;
const int concurrency = 16;

mutex m;
condition_variable cond;
atomic<bool> finished;

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

void anneal(int num, int k_max, double score_to_beat, double old_score) {
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
    time_t start_time = time(NULL);
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

    if (best_score < old_score) {
        cout << "NEW BEST SCORE (down from " << old_score << "): ";
        ofstream out("cpp-outputs/" + type + to_string(num) + ".out");
        for (int i = 0; i < nodes; i++) {
            out << best_x[i] << endl;
        }
        out.close();
        cout << best_score << " for input " << type << num << " with k_max = " << k_max << " (" << (time(NULL) - start_time) << " sec)" << endl;
    } else {
        // Move the line above to below the if/else if you want to see skip output
        // cout << "Skipping score ";
    }
}

void anneal_num(int num, double best_scores[]) {
    double score_to_beat = best_scores[input_offset + num - 1];
    int k_actual_max = max(2, (int) floor(2 * log(score_to_beat / 100.0)));
    int k_min = max(2, k_actual_max - 5);

    int w[nodes][nodes] = {};
    int x[nodes] = {};
    int k = get_k_from_output(num);
    int p[k] = {};
    unordered_set<int> old_s[k];
    double b[k] = {};
    double b_sum;
    double d;
    double old_score = 0;
    get_initial_state_from_output(
        num,
        k,
        w,
        x,
        p,
        old_s,
        b,
        &b_sum,
        &d,
        &old_score
    );

    if (!run_all && (old_score < score_to_beat || (abs(score_to_beat - old_score) < 0.001 && !try_to_break_ties))) {
        cout << "Already have a 1st place: " << old_score << " for input " << type << num << " with k_max = " << k << " (1st place is " << score_to_beat << ")" << endl;
    } else {
        cout << "Our current best for " << type << num << " is " << old_score << " (need to beat " << score_to_beat << ")" << endl;
        for (int k_max = k_min; k_max <= k_actual_max; k_max++) {
            anneal(num, k_max, score_to_beat, old_score);
        }
    }
    finished = true;
    cond.notify_all();
}

int main() {
    ifstream sfp("scores.txt");
    double best_scores[260 * 3];
    for (int i = 0; i < 260 * 3; i++) {
        sfp >> best_scores[i];
    }
    sfp.close();

    for (int i = 1; i <= 260; i++) {
        thread(anneal_num, i, best_scores).detach();
        if (i >= concurrency) {
            unique_lock<std::mutex> lock{m};
            cond.wait(lock, [&] {
                if (!finished) {
                    return false;
                } else {
                    finished = false;
                    return true;
                }
            });
        }
    }
    return 0;
}

// g++ -o rideThatSlay anneal.cpp -O3 -lpthread
