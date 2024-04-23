#include<iostream>
#include<random>
#include<cmath>
#include<numeric>
#include<vector>
#include<algorithm>
#include<limits>
#include<iomanip>

//setting basic problem parameters

constexpr int nsites {2}; //number of sites
constexpr int maxcars {21}; //number of states for each rental site
constexpr int rent_reward {10}; //money from a rent
constexpr int action_cost {-2}; //cost of moving a car

constexpr int lambdareturn[nsites] = {3,2}; //poisson parameters
constexpr int lambdarent[nsites] {3,4};

constexpr double gam {0.9}; //discount rate


//define a type for our policy iteration output - unused
class DP_couple{
public:
	int pol[maxcars*maxcars];
	double value[maxcars*maxcars];
};


//factorial - unused
int fact(int n){
	int output = 1;
	if (n == 0) return output;
	else{
		for (int i=2;i<=n;++i){
			output*=i;
		}
	return output;
	};
}


//define poisson probability, n number of cars
long double pois(int lambda, int n){
	
	long double ln_prob;
	long double probability;
	
	if (lambda==0){
		if (n==0)
			probability = 1;
		else probability = 0;
	} //lambda zero
	
	else {
		if (n==0)
			ln_prob = -lambda+n*std::log(lambda)-std::log(1);
		else {
			ln_prob = -lambda+n*std::log(lambda);
			for (int i=1;i<=n;++i){
				ln_prob -= std::log(i);
			}
		}
		probability = std::exp(ln_prob);
	} //positive lambda
	
	if (probability<0.000001) return 0.0;
	else return probability;
}


//cumulative poisson
long double cpois(int lambda, int n){
	long double cprobability {0.0};
	for (int i=0; i<=n; ++i){
		cprobability += pois(lambda, i);
	}
	return cprobability;
}


//skellam distribution for difference of poissons
long double skell(int lambda1, int lambda2, int n){
	long double p {1.0};
	
	if (lambda1 == lambda2){
		long double member = 2*lambda1;
		p *= std::cyl_bessel_i(std::abs(double(n)),member);
		p *= std::exp(-2*lambda1);
	}
	
	else{
	p =  std::pow(lambda1/double(lambda2),double(n)/2);
	long double member = 2*std::sqrt(lambda1*double(lambda2));
	p *= std::cyl_bessel_i(std::abs(double(n)),member);
	p *= std::exp(-(double)lambda1-(double)lambda2);
	}
	
	if (p>0.000001)
		return p;
	else 
		return 0.0; //prevent underflow
}


//cumulative skellam... there is a MAGIC CONSTANT HERE
//it should depend on lambdas really that 27.
long double cskell(int lambda1, int lambda2, int n){
	long double cprobability {0.0};
	for (int i=-25; i<=n; ++i){
		cprobability += skell(lambda1, lambda2, i);
	}
	return cprobability;
}


//uniform sampler - unused
int unif_sample(int min, int max)
{    
    std::random_device                  rand_dev;
    std::mt19937                        generator(rand_dev());
    std::uniform_int_distribution<int>  distr(min, max);
	
    return distr(generator);
}


//fills a vector with integers from min to max
std::vector<int> range_fill(int min, int max){
	
	std::vector<int> v;
	for (int i=min;i<=max;++i){
		v.push_back(i);
	}
	return v;
}


//expected reward
long double exp_reward(int state[nsites], int action){
	long double exprw {0.0};
	for (int i=0;i<nsites;++i){
		for (int j=0;j<state[i]; ++j){
			exprw += j*rent_reward*pois(lambdarent[i],j);
		}
		exprw += state[i]*(1-cpois(lambdarent[i],state[i])
			   + state[i]*pois(lambdarent[i],state[i]));
	}
	exprw += action_cost*std::abs(action);
	return exprw;
}


//state probability
long double nxtstate_p(int newstate[nsites], int oldstate[nsites], int action)
{
	int delta[nsites];
	delta[0] = newstate[0]-oldstate[0]+action; //action means how many cars i'm moving from 1 to 2
	delta[1] = newstate[1]-oldstate[1]-action; //defining only the random part of my problem
	
	long double probability[nsites];

	if (newstate[0] == 0)
	{ 
		//delta[0] = 0 - oldstate[0];
		probability[0] = cskell(lambdareturn[0],lambdarent[0],delta[0]);
	}
	else if (newstate[0] == 20)
	{
		//delta[0] = 20 - oldstate[0];
		probability[0] = 1-cskell(lambdareturn[0],lambdarent[0],delta[0])
					   + skell(lambdareturn[0],lambdarent[0],delta[0]);
	}
	else probability[0] = skell(lambdareturn[0],lambdarent[0],delta[0]);
	
	if (newstate[1] == 0)
	{	
		//delta[1] = 0 - oldstate[1];
		probability[1] = cskell(lambdareturn[1],lambdarent[1],delta[1]);
	}
	else if (newstate[1] == 20)
	{	
		//delta[1] = 20 - oldstate[1];
		probability[1] = 1-cskell(lambdareturn[1],lambdarent[1],delta[1])
					   + skell(lambdareturn[1],lambdarent[1],delta[1]); 
	}
	else probability[1] = skell(lambdareturn[1],lambdarent[1],delta[1]);

	
	//newstates are independent given the action
	long double jprobability = probability[0]*probability[1];
	return jprobability;
}


//policy evaluation
void policy_evaluation(double theta, 
					   long double values[maxcars*maxcars], int policy[maxcars*maxcars],
					   int states[maxcars*maxcars][nsites])
{	
	int count = 0;
	long double v;
	long double delta {0.0};
	long double newv;
	do{
		++count;
		delta = 0.0;
		for (int i=0;i<maxcars*maxcars;++i){
			
			v = values[i];
			newv = 0;
			int state[nsites] = {states[i][0],states[i][1]};//state
			
			long double exp_rw = exp_reward(state,policy[i]);
			newv += exp_rw;
			
			for (int j=0;j<maxcars*maxcars;++j){
				int newstate[nsites] = {states[j][0],states[j][1]};
				long double probability = nxtstate_p(newstate,state,policy[i]);
				newv += values[j]*gam*probability;
			}
			values[i] = newv;
			delta = std::max(delta,std::abs(v-values[i]));
		}
		if (count%2==0)
			std::cout << "Policy evaluation loop " << count << ": " << delta << ".\n";
	} while(!(delta<theta));
}


//policy improvement
void policy_improvement(int policy[maxcars*maxcars], long double values[maxcars*maxcars],
						int states[maxcars*maxcars][nsites], int actions[maxcars*maxcars][nsites])
{
	for (int i=0;i<maxcars*maxcars;++i){
		
		//int p = policy[i];
		int state[nsites] = {states[i][0],states[i][1]}; //state
		std::vector<int> action = range_fill(actions[i][0],actions[i][1]);//action set at state
		
		long double maximum = std::numeric_limits<double>::min();
		int best_action;
		long double criterion {0.0};
		for(int a : action){
			
			criterion = 0.0;
			long double exp_rw = exp_reward(state,a);
			criterion += exp_rw;
			
			for (int j=0;j<maxcars*maxcars;++j){
				int newstate[nsites] = {states[j][0],states[j][1]};
				long double probability = nxtstate_p(newstate,state,a);
				criterion += values[j]*gam*probability;
			}
			
			if (criterion>=maximum){
				best_action = a;
				maximum = criterion;
			}
			
		} //this time we need an extra loop, to loop through actions
		
		policy[i] = best_action;
	}
	std::cout << "Policy updated!\n";
}


//max pointer
long double* max_pointer(long double values[maxcars*maxcars]){
	long double* max;
	long double cmax = std::numeric_limits<double>::min();
	for (int i=0;i<maxcars*maxcars;++i){
		if (values[i]>=cmax) max = &values[i];
	}
	return max;
}


//max value
long double max_el(long double values[maxcars*maxcars]){
	long double max;
	long double cmax = std::numeric_limits<double>::min();
	for (int i=0;i<maxcars*maxcars;++i){
		if (values[i]>=cmax) max = values[i];
	}
	return max;
}


//policy iteration algorithm
void policy_iteration(double theta, double eta,
					  int policy[maxcars*maxcars], long double values[maxcars*maxcars],
					  int states[maxcars*maxcars][nsites], int actions[maxcars*maxcars][nsites])
{	// this is the same as delta, just different name
	long double gap = std::numeric_limits<double>::max();
	long double v;
	long double v_up;
	long double* old_max;
	long double* new_max;
	bool flag;
	int count = 0;
	do{
		++count;
		flag=false;
		old_max = std::max_element(values,values+maxcars*maxcars);
		v = *old_max;
		
		policy_evaluation(theta,values,policy,states);
		policy_improvement(policy,values,states,actions);
		
		new_max = std::max_element(values,values+maxcars*maxcars);

		flag = (old_max == new_max);
		if (flag){
			v_up = *new_max;
			gap = std::min(gap,std::abs(v_up-v));
		}
		if (count%3 == 0)
			std::cout<<"Policy iteration loop " << count << ": " << gap << ".\n";
	} while(!(gap<eta & flag==true)); //stop when the maximumum difference
}


int main()
{   
	//states
	int states[maxcars*maxcars][nsites];
	
	//fill action-states matrix
	for (int j=0;j<maxcars;++j){
		for (int i=0;i<maxcars;++i){
			states[j*maxcars+i][0] = j;
			states[j*maxcars+i][1] = i;
		}
	}

	//actions
	int actions[maxcars*maxcars][nsites];

	//build a matrix to define the action set for each state
	for (int j=0;j<maxcars;++j){
		for (int i=0;i<maxcars;++i){
			actions[j*maxcars+i][0] = -std::min(5,i); //if i have less than 5 cars in site 2, i can't move 5 cars to 1
			actions[j*maxcars+i][1] = std::min(5,j); //if i have less than 5 cars in site 1, i can't move 5 cars to 2
		}
	}
	
	//initialize policy and values;
	int policy[maxcars*maxcars] = {0};
	long double values[maxcars*maxcars] = {0.0};
	//setting parameters
	double theta = 0.0001;
	double eta = 0.0001;
	
	//running policy iteration
	policy_iteration(theta, eta,
					 policy, values,
					 states, actions);
	
	//create a table to visualize the policy
	long double policy_table[maxcars][maxcars];
	for (int i=0;i<=maxcars*maxcars;++i){
		policy_table[states[i][0]][states[i][1]] = policy[i];
	}
	for(int i=0;i<maxcars;i++){
	  	for(int j=0;j<maxcars;j++)
	  	{
	  		std::cout<<" "<< std::setw(10) << policy_table[i][j];
		}
		std::cout<<std::endl; 
	}
	
	return 0;
}




