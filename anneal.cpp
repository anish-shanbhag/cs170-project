#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <time.h>
#include <pthread.h>

using namespace std;
#define NUM_THREADS 32

struct thread_data {
   int  thread_id;
   int k_max;
   char *message;
};

void *anneal(void *tdObj) {
    const double e = 2.718281828459;
    double T_min = 4.5;
    double T_max = 33000;
    int steps = 5000;

    const int size = 100;
    const int k_max = ((thread_data *)tdObj)->k_max;

    int w[size][size];

    int theNum = ((thread_data *)tdObj)->thread_id;
    string name = "small" + to_string(theNum);

    ifstream fp("weights/" + name + ".txt");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fp >> w[i][j];
        }
    }

    // state variables
    int x[size];
    int p[k_max];
    for (int i = 0; i < k_max; i++) {
        p[i] = 0;
    }
    double b[k_max];
    for (int i = 0; i < k_max; i++) {
        b[i] = 0;
    }
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

    cout << "Best assignment (" << best_score << ") " << name << " : " << endl << "[" << best_x[0];
    for (int i = 1; i < size; i++) {
        cout << ", " << best_x[i];
    }
    cout << "]" << endl;


    ofstream out(name + "_" + to_string(k_max) + ".out");
    out << "[" << best_x[0];
    for (int i = 1; i < size; i++) {
        out << ", " << best_x[i];
    }
    out << "]" << endl;
    out.close();

   pthread_exit(NULL);
}

int main() {
    int num = 6;
    int k_min = 7;
    int k_max = 7;
    pthread_t threads[k_max - k_min];
    struct thread_data td[k_max - k_min];
    int rc;
    int i;
    for( i = k_min; i <= k_max; i++ ) {
      cout <<"main() : creating thread, " << i << endl;
      td[i - k_min].thread_id = num;
      td[i - k_min].message = "This is message";
      td[i - k_min].k_max = i;
      rc = pthread_create(&threads[i - k_min], NULL, anneal, (void *)&td[i - k_min]);

      if (rc) {
         cout << "Error:unable to create thread," << rc << endl;
         exit(-1);
      }
   }
   pthread_exit(NULL);
}

// g++ -o rideThatSlay anneal.cpp -O3 -funroll-loops -lpthread
