/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::default_random_engine;
using std::numeric_limits;
using std::discrete_distribution;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i)
  {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  for (auto& p : particles) {
    // Previous position and heading of the particle
    double x_0 = p.x;
    double y_0 = p.y;
    double theta_0 = p.theta;

    // Use the motion model to predict where the particle will be at the next
    // time step
    double x_f, y_f, theta_f;
    if (abs(yaw_rate) < 1e-3) {
      theta_f = theta_0;
      x_f = x_0 + velocity * delta_t * cos(theta_0);
      y_f = y_0 + velocity * delta_t * sin(theta_0);
    } else {
      theta_f = theta_0 + yaw_rate * delta_t;
      x_f = x_0 + velocity * (sin(theta_f) - sin(theta_0)) / yaw_rate;
      y_f = y_0 + velocity * (cos(theta_0) - cos(theta_f)) / yaw_rate;
    }

    // Create normal distributions centered on predicted values
    normal_distribution<double> dist_x(x_f, std_pos[0]);
    normal_distribution<double> dist_y(y_f, std_pos[1]);
    normal_distribution<double> dist_theta(theta_f, std_pos[2]);

    // Realize normal distributions
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (auto& obs : observations) {
    double closest = numeric_limits<double>::max();
    for (const auto& pre : predicted) {
      double distance = dist(obs.x, obs.y, pre.x, pre.y);
      if (distance < closest) {
        obs.id = pre.id;
        closest = distance;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  const double std_x = std_landmark[0];
  const double std_y = std_landmark[1];
  double all_weights = 0;

  for (auto& p : particles) {
    vector<LandmarkObs> predictions;
    for (const auto& l : map_landmarks.landmark_list) {
      double distance = dist(p.x, p.y, l.x_f, l.y_f);
      if (distance < sensor_range) {
        LandmarkObs prediction;
        prediction.id = l.id_i;
        prediction.x = l.x_f;
        prediction.y = l.y_f;
        predictions.push_back(prediction);
      }
    }

    vector<LandmarkObs> trans_observations(observations);
    for (auto& obs : trans_observations) {
      double x_m = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
      double y_m = p.y + (sin(p.theta) * obs.x) - (cos(p.theta) * obs.y);
      obs.x = x_m;
      obs.y = y_m;
    }

    dataAssociation(predictions, trans_observations);

    double weight = 1;
    for (const auto& obs : trans_observations) {
      double mu_x, mu_y;
      for (const auto& pre : predictions) {
        if (pre.id == obs.id) {
          mu_x = pre.x;
          mu_y = pre.y;
          break;
        }
      }
      weight *= multiv_prob(std_x, std_y, obs.x, obs.y, mu_x, mu_y);
    }

    p.weight = weight;
    all_weights += weight;
  }

  for (auto& p : particles) {
    p.weight /= all_weights;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<double> weights;
  for (const auto& p : particles) {
    weights.push_back(p.weight);
  }
  
  discrete_distribution<int> dist(weights.begin(), weights.end());
  
  vector<Particle> resampled;
  for (int i = 0; i < num_particles; ++i) {
    int j = dist(gen);
    resampled.push_back(particles[j]);
  }

  particles = resampled;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}