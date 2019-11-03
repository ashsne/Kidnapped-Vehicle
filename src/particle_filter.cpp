#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using namespace std;

// Create only once the default random engine
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles
  num_particles = 1000;

	// Set x, y and theta with standard deviations
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for(int i = 0; i < num_particles; ++i){
		Particle p;
		p.id 			= i;
		p.x 			= dist_x(gen);
		p.y 			= dist_y(gen);
		p.theta 		= dist_theta(gen);
		p.weight 		= 1.0;

		// Add this particle to particles set
		particles.push_back(p);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {

	double x0, y0, theta0;
	double x_pred, y_pred, theta_pred;

	for(int i=0; i < num_particles; ++i){

		// Initial position and heading direction
		x0 = particles[i].x;
		y0 = particles[i].y;
		theta0 = particles[i].theta;

		if(abs(yaw_rate) > 1e-5){
			x_pred =  x0 + velocity/yaw_rate * (sin(theta0 + yaw_rate*delta_t) - sin(theta0));
			y_pred =  y0 + velocity/yaw_rate * (-cos(theta0 + yaw_rate*delta_t) + cos(theta0));
			theta_pred = theta0 + yaw_rate*delta_t;
		} else {
			x_pred = x0 + velocity * delta_t * cos(theta0);
			y_pred = y0 + velocity * delta_t * sin(theta0);
			theta_pred = theta0;

		}

		// Get the normal distribution of the predicted state
		normal_distribution<double> dist_x(x_pred, std_pos[0]);
		normal_distribution<double> dist_y(y_pred, std_pos[1]);
		normal_distribution<double> dist_theta(theta_pred, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {

	for (auto& obs : observations){
		double min_dist = std::numeric_limits<int>::max(); // Initialize with infinity

		for(const auto& pred_obs : predicted){

			// Get the distance between predicted and observation
			double d = dist(pred_obs.x, pred_obs.y, obs.x, obs.y);

			// Update the observation ID if the new distance is lower than the previous one
			if(d < min_dist){
				obs.id = pred_obs.id;
				min_dist = d;
			}

		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

	// Go through all the particles
	for(int i = 0; i < num_particles; ++i){

		// Get the position and orientation of the ith particle
		double p_x, p_y, p_theta;
		p_x = particles[i].x;
		p_y = particles[i].y;
		p_theta = particles[i].theta;

		// Transform observations (these are measured wrt particle) into map coordinates
		vector<LandmarkObs> landmark_observation_map_ref;

		for (const auto& observation : observations){

			double x_c = observation.x;
			double y_c = observation.y;
			LandmarkObs coord_map_ref;

			coord_map_ref.x = p_x + cos(p_theta) * x_c - sin(p_theta) * y_c;
			coord_map_ref.y = p_y + sin(p_theta) * x_c + cos(p_theta) * y_c;
			coord_map_ref.id = observation.id;

			// Add this transformed coordinates of the observed landmark to the mapped system
			landmark_observation_map_ref.push_back(coord_map_ref);
		}

		// Select all the landmark positions within the sensor range
		vector<LandmarkObs> predicted_landmarks;

		for (const auto& map_landmark : map_landmarks.landmark_list) {

            int l_id   = map_landmark.id_i;
            double l_x = (double) map_landmark.x_f;
            double l_y = (double) map_landmark.y_f;

            double d = dist(p_x, p_y, l_x, l_y);
            if (d < sensor_range) {
                LandmarkObs l_pred;
                l_pred.id = l_id;
                l_pred.x = l_x;
                l_pred.y = l_y;
                predicted_landmarks.push_back(l_pred);
            }
		}
		// Data association: Identify which obrevations corrospond to the predicted_landmarks
		dataAssociation(predicted_landmarks, landmark_observation_map_ref);

		// Calculate the likelihood probability of this particle
		double particle_likelihood = 1.0;
		double mu_x, mu_y;

		double std_x = std_landmark[0];
		double std_y = std_landmark[1];

		// loop over all the observations
		for (const auto& land_obs : landmark_observation_map_ref){

			// Find out the corrosponding landmark to the curretn obervation
			for(const auto& pred_land : predicted_landmarks) {
				if (land_obs.id == pred_land.id){
					mu_x = pred_land.x;
					mu_y = pred_land.y;
					break;
				}
			}

			// Calculate the gaussian probability
			double norm_factor = 1/ (2 * M_PI * std_x * std_y);
			double prob = exp( -(0.5 * pow(((land_obs.x - mu_x)/std_x), 2) + 0.5 * pow(((land_obs.y - mu_y)/std_y), 2)));

			particle_likelihood *= norm_factor * prob;
		}
		particles[i].weight = particle_likelihood;
	} // end loop for the each particle likelihood

	// Normalize the particle likelihood so that summation of them all will be 1

	// Get the summation of all the probabilities
	double norm_factor_total = 0.0;
	for (const auto& particle : particles){
		norm_factor_total += particle.weight;
	}
	// Normalization loop
	for(auto& particle : particles){
		particle.weight /= (norm_factor_total + numeric_limits<double>::epsilon());
	}
}

void ParticleFilter::resample() {

	// Get a weights vector from all the particles
	vector<double> weights_vec;
	for(const auto& particle : particles){
		weights_vec.push_back(particle.weight);
	}

	// Get a dicrete distribution depending on the corrosponding weights
	discrete_distribution<int> weighted_distribution(weights_vec.begin(), weights_vec.end());

	// Resample the particles wrt weights
	vector<Particle> resampled_particles;
	for (size_t i = 0; i < num_particles; ++i) {
			int k = weighted_distribution(gen);
			resampled_particles.push_back(particles[k]);
	}

	particles = resampled_particles;

	// Reset weights for all particles
	for (auto& particle : particles)
			particle.weight = 1.0;

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
