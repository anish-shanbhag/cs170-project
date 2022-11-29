#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <time.h>

using namespace std;

const double e = 2.718281828459;

int main() {
    double T_min = 4.5;
    double T_max = 33000;
    int steps = 100000000;

    const int size = 100;
    const int k_max = 12;

    int w[size][size];

    string name = "small34";

    ifstream fp("weights/" + name + ".txt");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fp >> w[i][j];
        }
    }

    // state variables
    int x[size];
    int p[k_max] = {};
    double b[k_max] = {};
    double b_sum = 0;
    unordered_set<int> s[k_max];
    double d;
    double score = 0;

    // initial state
    for (int i = 0; i < size; i++) {
        x[i] = (i % k_max) + 1;
        p[x[i] - 1]++;
        s[x[i] - 1].insert(i);
    }

    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            if (x[i] == x[j]) {
                score += w[i][j];
            }
        }
    }

    score += 100 * pow(e, 0.5 * k_max);

    for (int i = 0; i < k_max; i++) {
        b[i] = (double) p[i] / size - 1.0 / k_max;
    }
    for (int i = 0; i < k_max; i++) {
        b_sum += b[i] * b[i];
    }
    d = pow(e, 70 * sqrt(b_sum));
    score += d;


    // simulated annealing
    clock_t start_time = clock();
    double best_score = 1000000000;
    double best_x[size];
    double T = T_max;
    double T_factor = -log(T_max / T_min);
    random_device dev;
    mt19937 rng(dev());
    uniform_int_distribution<mt19937::result_type> x_dist(0, size - 1);
    uniform_int_distribution<mt19937::result_type> k_dist(1, k_max);
    uniform_real_distribution<double> T_dist(0.0, 1.0);

    for (double step = 0; step < steps; step++) {
        T = T_max * pow(e, T_factor * step / steps);
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

        double new_b_sum = b_sum - pow(b[x[i] - 1], 2) - pow(b[new_x - 1], 2) + pow(b[x[i] - 1] - 1.0 / size, 2) + pow(b[new_x - 1] + 1.0 / size, 2);
        double new_d = pow(e, 70 * sqrt(new_b_sum));
        delta += new_d - d;

        if (delta <= 0 || exp(-delta / T) > T_dist(rng)) {
            score += delta;
            b_sum = new_b_sum;
            d = new_d;
            p[x[i] - 1]--;
            p[new_x - 1]++;
            b[x[i] - 1] -= 1.0 / size;
            b[new_x - 1] += 1.0 / size;
            s[x[i] - 1].erase(i);
            s[new_x - 1].insert(i);
            x[i] = new_x;

            if (score < best_score) {
                best_score = score;
                for (int i = 0; i < size; i++) {
                    best_x[i] = x[i];
                }
                cout << "step " << step << " score " << score << endl;
            }
        }
    }

    cout << "Best assignment: " << endl << "[" << best_x[0];
    for (int i = 1; i < size; i++) {
        cout << ", " << best_x[i];
    }
    cout << "]" << endl;

    
    ofstream out(name + ".out");
    out << "[" << best_x[0];
    for (int i = 1; i < size; i++) {
        out << ", " << best_x[i];
    }
    out << "]" << endl;
    out.close();


    cout << "Finished " << steps << " steps in " << (clock() - start_time) / (double) CLOCKS_PER_SEC << " seconds" << endl;

    return 0;
}
