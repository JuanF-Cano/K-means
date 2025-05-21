#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>

using namespace std;

// Class representing a data point with features and assigned cluster ID
class Point {
  public:
    vector<double> features;  // Feature vector
    int cluster_id;           // Assigned cluster index

    Point(const vector<double> &f) : features(f), cluster_id(-1) {}
};

// KMeans clustering implementation
class KMeans {
  private:
    int k;                             // Number of clusters
    int max_iters;                     // Maximum number of iterations
    vector<vector<double>> centroids; // Centroids of clusters

    // Calculate Euclidean distance between two feature vectors
    double euclideanDistance(const vector<double> &a, const vector<double> &b) {
      double sum = 0.0;
      for (size_t i = 0; i < a.size(); ++i)
        sum += pow(a[i] - b[i], 2);
      return sqrt(sum);
    }

    // Assign each point to the closest centroid's cluster
    void assignClusters(vector<Point> &points) {
      for (auto &point : points) {
        double min_dist = numeric_limits<double>::max();
        int best_cluster = -1;
        for (int i = 0; i < k; ++i) {
          double dist = euclideanDistance(point.features, centroids[i]);
          if (dist < min_dist) {
            min_dist = dist;
            best_cluster = i;
          }
        }
        point.cluster_id = best_cluster;
      }
    }

    // Recalculate centroids as mean of points in each cluster
    void updateCentroids(vector<Point> &points) {
      vector<vector<double>> new_centroids(k, vector<double>(points[0].features.size(), 0.0));
      vector<int> counts(k, 0);

      for (const auto &point : points) {
        for (size_t j = 0; j < point.features.size(); ++j)
          new_centroids[point.cluster_id][j] += point.features[j];
        counts[point.cluster_id]++;
      }

      for (int i = 0; i < k; ++i) {
        if (counts[i] == 0) continue; // Avoid division by zero
        for (size_t j = 0; j < new_centroids[i].size(); ++j) 
          new_centroids[i][j] /= counts[i];
      }

      centroids = new_centroids;
    }

  public:
    // Constructor to initialize k and maximum iterations
    KMeans(int k, int max_iters) : k(k), max_iters(max_iters) {}

    // Getter for centroids after training
    const vector<vector<double>>& getCentroids() const { return centroids; }

    // Run the KMeans clustering algorithm on the dataset
    void fit(vector<Point> &points) {
      centroids.clear();
      srand(time(0));
      // Initialize centroids randomly from points
      for (int i = 0; i < k; ++i) {
        int index = rand() % points.size();
        centroids.push_back(points[index].features);
      }

      // Main iterative loop
      for (int iter = 0; iter < max_iters; ++iter) {
        assignClusters(points);

        vector<vector<double>> old_centroids = centroids;
        updateCentroids(points);

        // Compute total centroid movement to check for convergence
        double change = 0.0;
        for (int i = 0; i < k; ++i)
            change += euclideanDistance(old_centroids[i], centroids[i]);

        if (change < 1e-4) break; // Stop if centroids stabilize
      }
    }

    // Print clustering results and statistics to console
    void printResults(const vector<Point> &points) {
      vector<int> cluster_counts(k, 0);

      for (const auto &point : points) {
        cluster_counts[point.cluster_id]++;
        cout << "Cluster " << point.cluster_id << ": ";
        for (double val : point.features) cout << val << " ";
        cout << endl;
      }

      cout << "\n--- Summary by Cluster ---\n";
      for (int i = 0; i < k; ++i) {
        cout << "Cluster " << i << ": " << cluster_counts[i] << " elements\n";
      }

      cout << "\n--- Final Centroids ---\n";
      for (int i = 0; i < k; ++i) {
        cout << "Cluster " << i << " centroid: ";
        for (double val : centroids[i])
          cout << val << " ";
        cout << endl;
      }
    }
};

// Load data points from a tab-separated file with specific column selection
vector<Point> loadData(const string &filename) {
  ifstream file(filename);
  string line;
  vector<Point> data;

  bool header_skipped = false;

  while (getline(file, line)) {
    if (!header_skipped) {
      header_skipped = true; // Skip header line
      continue;
    }

    stringstream ss(line);
    string cell;
    vector<double> features;
    int col = 0;

    // Extract selected columns: column 4 and columns 9 to 14 (0-based)
    while (getline(ss, cell, '\t')) {
      if (col == 4 || (col >= 9 && col <= 14)) {
        try {
          double val = cell.empty() ? 0.0 : stod(cell);
          features.push_back(val);
        }
        catch (...) {
          features.push_back(0.0); // Use zero if conversion fails
        }
      }
      col++;
    }

    if (!features.empty()) {
      data.emplace_back(features);
    }
  }

  return data;
}

// Normalize data features to range [0,1] per feature dimension
void normalizeData(vector<Point> &data) {
  if (data.empty()) return;

  size_t dim = data[0].features.size();
  vector<double> min_vals(dim, numeric_limits<double>::max());
  vector<double> max_vals(dim, numeric_limits<double>::lowest());

  // Find min and max per feature
  for (const auto &p : data) {
    for (size_t i = 0; i < dim; ++i) {
      min_vals[i] = min(min_vals[i], p.features[i]);
      max_vals[i] = max(max_vals[i], p.features[i]);
    }
  }

  // Scale features to [0,1]
  for (auto &p : data) {
    for (size_t i = 0; i < dim; ++i) {
      if (max_vals[i] != min_vals[i])
        p.features[i] = (p.features[i] - min_vals[i]) / (max_vals[i] - min_vals[i]);
    }
  }
}

// Save points with their cluster assignments to CSV file
void saveClusters(const vector<Point>& points, const string& filename) {
  ofstream file(filename);
  for (const auto& p : points) {
    for (double f : p.features)
      file << f << ",";
    file << p.cluster_id << "\n";
  }
  file.close();
}

// Save cluster centroids to CSV file
void saveCentroids(const vector<vector<double>> &centroids, const string &filename) {
  ofstream file(filename);
  for (const auto& centroid : centroids) {
    for (double val : centroid)
      file << val << ",";
    file << "\n";
  }
  file.close();
}

int main() {
  // Load data from file
  vector<Point> data = loadData("kmeans_cleaned.txt");
  if (data.empty()) {
    cerr << "No data loaded. Check the input file.\n";
    return 1;
  }

  // Normalize feature values
  normalizeData(data);

  // Initialize and run KMeans with 3 clusters and max 100 iterations
  KMeans kmeans(3, 100);
  kmeans.fit(data);

  // Print clustering results
  kmeans.printResults(data);

  // Save clustered data and centroids to files
  saveClusters(data, "clustered_data.csv");
  saveCentroids(kmeans.getCentroids(), "centroids.csv");

  return 0;
}