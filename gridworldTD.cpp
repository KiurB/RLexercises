#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>
#include <array>
using namespace std;

template <typename T, size_t Dim1, size_t Dim2, size_t Dim3, size_t Dim4>
using Array4d = array<array<array<array<T,Dim4>,Dim3>,Dim2>,Dim1>;

template <typename T, size_t Dim1, size_t Dim2>
using Array2d = array<array<T,Dim2>,Dim1>;
/* Here I want to solve some gridworld problem with some wind.*/

// I set the parameters

constexpr int grid_wdt {10}; // I add padding
constexpr int grid_hgt {7}; // I add padding
constexpr int n_dir {3};
constexpr int n_dim {2};
constexpr int wind_power {1};

constexpr int g_reward {0};
constexpr int n_reward {-1};

// Sampler

int unif_sampler(int min, int max)
{
	random_device			rand_dev;
	mt19937				generator(rand_dev());
	uniform_int_distribution<int>	distr(min, max);
	
	return distr(generator);
}

// Action

array<int,n_dim> max_a(long double action_value[grid_wdt][grid_hgt][n_dir][n_dir],
	      	       array<int,n_dim> state)
{
	double max = numeric_limits<double>::min();
	array<int,n_dim> index;
	for (int k=0;k<n_dir;++k){
        for (int h=0;h<n_dir;++h){
      		if (action_value[state[0]][state[1]][k][h]>=max){
			index[0] = k;
			index[1] = h;
			max = action_value[state[0]][state[1]][k][h];
			}
      	}}
	return index;
}

// I use an e-greedy policy... I'm not choosing randomly from maximizing actions but oke
array<int,n_dim> policy(long double action_value[grid_wdt][grid_hgt][n_dir][n_dir],
			int e, array<int,n_dim> state)
{
	// coin toss
	int coin = unif_sampler(0,100);
	array<int,n_dim> action;
	
	if (coin <= 100-e){	
	action = max_a(action_value,state);
	action[0] -= 1;
	action[1] -= 1; // I want actions to be -1 0 1, not 0 1 2
	}
	else {
	action[0] = unif_sampler(-1,1);
	action[1] = unif_sampler(-1,1);
	}
	
	return action;
}

// Main - SARSA

int main()
{
	// parameters
	double alpha = 0.5;
	int e = 5;
	int n_episoded = 200;
	array<int,n_dim> goal = {6,4};
	int wind = 2;
	double gamma = 0.8;
	// state-action value grid
	long double action_value[grid_wdt][grid_hgt][n_dir][n_dir] = {0.};
	for (int i=0;i<grid_wdt;++i){
	for (int j=0;j<grid_hgt;++j){
        for (int k=0;k<n_dir;++k){
        for (int h=0;h<n_dir;++h){
      		action_value[i][j][k][h] = 2.;
        }}}}
        for (int i=0;i<n_dir;++i){
	for (int j=0;j<n_dir;++j){
		action_value[goal[0]][goal[1]][i][j] = 0.;
	}}
	
	// Main loop
	for (int i=0;i<n_episoded;++i)
	{	
	
	// state
	array<int,n_dim> state;
	state[0] = unif_sampler(0, grid_wdt-1);
	state[1] = unif_sampler(0, grid_hgt-1);
	// choose action
	array<int,n_dim> a = policy(action_value,e,state);
	
	// Inner loop
	while (state != goal)
	{
		// Update state and get reward
		array<int,n_dim> nstate = {0};
		nstate[0] = state[0] + a[0];
		nstate[1] = state[1] + a[1];
		if (nstate[0] == wind) nstate[0] += wind_power;
		// If out of bounds, return to start
		if (nstate[0]>=grid_wdt || nstate[1]>=grid_hgt){
		nstate[0] = unif_sampler(0, grid_wdt-1);
		nstate[1] = unif_sampler(0, grid_hgt-1);
		}		
		if (nstate[0]<0 || nstate[1]<0){
		nstate[0] = unif_sampler(0, grid_wdt-1);
		nstate[1] = unif_sampler(0, grid_hgt-1);
		}

		int reward;
		if (nstate == goal) reward = g_reward;
		else reward = n_reward;

		// Select action
		array<int,n_dim> na = policy(action_value,e,nstate);
		action_value[state[0]][state[1]][a[0]+1][a[1]+1] += 
			alpha*(reward+gamma*action_value[nstate[0]][nstate[1]][na[0]+1][na[1]+1]
			-action_value[state[0]][state[1]][a[0]+1][a[1]+1]);
		state[0] = nstate[0];
		state[1] = nstate[1];
		a[0] = na[0];
		a[1] = na[1];
		}
	}
	
	long double value[grid_wdt][grid_hgt] = {0};
	for (int i=0;i<grid_wdt;++i){
	for (int j=0;j<grid_hgt;++j){
        for (int k=0;k<n_dir;++k){
        for (int h=0;h<n_dir;++h){
      		value[i][j] += action_value[i][j][k][h];
        }}
		value[i][j] /= n_dir*n_dir;
	}}
	
	//visualize the value
	for(int i=0;i<grid_wdt;i++){
	  	for(int j=0;j<grid_hgt;j++)
	  	{
	  		std::cout<<" "<< std::setw(10) << value[i][j];
		}
		std::cout<<std::endl; 
	}

	return 0;
}
