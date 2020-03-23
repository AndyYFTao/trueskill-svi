using Revise # lets you change A2funcs without restarting julia!
include("A2_src.jl")
using Plots
using Statistics: mean
using Zygote
using Test
using Logging
using .A2funcs: log1pexp # log(1 + exp(x)) stable
using .A2funcs: factorized_gaussian_log_density
using .A2funcs: skillcontour!
using .A2funcs: plot_line_equal_skill!

function log_prior(zs)
  # Note the transpose to match the size
  # Input: a (N,K) array. K sets of skills for N players.
  # Handouts Output: a K * 1 array. Each row: the log-prior for that set of skills.
  # Code Output: size == (1 * K)
  return  factorized_gaussian_log_density(0,0,zs)
end

function logp_a_beats_b(za,zb)
  # To match dimensions
  z_a = za'
  z_b = zb'
  return -log1pexp.(z_b - z_a)
end


function all_games_log_likelihood(zs,games)
  # Input: size(zs) == [N,K] & size(games) == (M,2)
  # M [games/winners/losers], K [skills/batch_size]
  z_s = zs'
  zs_a = z_s[:, games[:,1]] # size == (K,M) for winners
  zs_b = z_s[:, games[:,2]] # size == (K,M) for losers
  likelihoods =  sum(logp_a_beats_b(zs_a,zs_b), dims=1) # Code size == (1,K)
  return likelihoods
end

function joint_log_density(zs,games)
  # Input: size(zs) == [N,K] & size(games) == (M,2)
  # Code Output: size == (1,K)
  return log_prior(zs) + all_games_log_likelihood(zs,games)
end

@testset "Test shapes of batches for likelihoods" begin
  B = 15 # number of elements in batch
  N = 4 # Total Number of Players
  test_zs = randn(4,15)
  test_games = [1 2; 3 1; 4 2] # 1 beat 2, 3 beat 1, 4 beat 2
  @test size(test_zs) == (N,B)
  #batch of priors
  @test size(log_prior(test_zs)) == (1,B)
  # loglikelihood of p1 beat p2 for first sample in batch
  @test size(logp_a_beats_b(test_zs[1,1],test_zs[2,1])) == ()
  # loglikelihood of p1 beat p2 broadcasted over whole batch
  @test size(logp_a_beats_b.(test_zs[1,:],test_zs[2,:])) == (B,)
  # batch loglikelihood for evidence
  @test size(all_games_log_likelihood(test_zs,test_games)) == (1,B)
  # batch loglikelihood under joint of evidence and prior
  @test size(joint_log_density(test_zs,test_games)) == (1,B)
end

# Convenience function for producing toy games between two players.
two_player_toy_games(p1_wins, p2_wins) = vcat([repeat([1,2]',p1_wins), repeat([2,1]',p2_wins)]...)

# Example for how to use contour plotting code
plot(title="Example Gaussian Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
example_gaussian(zs) = exp(factorized_gaussian_log_density([-1.,2.],[0.,0.5],zs))
skillcontour!(example_gaussian) # label="example gaussian"
plot_line_equal_skill!()
savefig(joinpath("plots","example_gaussian.pdf"))


# TODO: plot prior contours
plot(title="Prior Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
joint_prior(zs) = exp(log_prior(zs))
skillcontour!(joint_prior) # label="joint prior"
plot_line_equal_skill!()
savefig(joinpath("plots","joint_prior.png"))


# TODO: plot likelihood contours
plot(title="Likelihood Contour Plot",
  xlabel = "Player 1 Skill",
  ylabel = "Player 2 Skill"
   )
# e.g. player A wins 5 games and player B wins 3 games
games_5_3 = two_player_toy_games(5,3)
# multivariate --> univariate
all_games_likelihood(zs) = exp.(all_games_log_likelihood(zs,games_5_3))
skillcontour!(all_games_likelihood) # label="likelihood"
plot_line_equal_skill!()
savefig(joinpath("plots","likelihood.png"))



# TODO: plot joint contours with player A winning 1 game
plot(title="Joint Contour Plot with A winning 1 game",
  xlabel = "Player 1 Skill",
  ylabel = "Player 2 Skill"
   )
games_1_0 = two_player_toy_games(1,0)
# multivariate --> univariate
join_density(zs) = exp.(joint_log_density(zs, games_1_0))
skillcontour!(join_density) # label="join_density_w_A1B0"
plot_line_equal_skill!()
savefig(joinpath("plots","joint_a1b0.png"))



# TODO: plot joint contours with player A winning 10 games
plot(title="Joint Contour Plot with A winning 10 games",
  xlabel = "Player 1 Skill",
  ylabel = "Player 2 Skill"
   )
games_10_0 = two_player_toy_games(10,0)
# multivariate --> univariate
join_density_10(zs) = exp.(joint_log_density(zs, games_10_0))
skillcontour!(join_density_10) # label="join_density_w_A10B0"
plot_line_equal_skill!()
savefig(joinpath("plots","joint_a10b0.png"))


#TODO: plot joint contours with player A winning 10 games and player B winning 10 games
plot(title="Joint Contour Plot with A:B = 10:10",
  xlabel = "Player 1 Skill",
  ylabel = "Player 2 Skill"
   )
games_10_10 = two_player_toy_games(10,10)
# multivariate --> univariate
join_density_10_10(zs) = exp.(joint_log_density(zs, games_10_10))
skillcontour!(join_density_10_10) # label="join_density_w_A10B10"
plot_line_equal_skill!()
savefig(joinpath("plots","joint_a10b10.png"))



function elbo(params,logp,num_samples)
  mu = params[1] # size = num_players
  logsig = params[2]
  mu_ = vcat([repeat(mu',num_samples)]...)' # size = (num_players,num_samples)
  stdev_ = exp.(vcat([repeat(logsig',num_samples)]...)')
  num_players = size(mu)[1]
  samples = randn(num_players, num_samples) .* stdev_ + mu_ #reparameterization trick
  logp_estimate = logp(samples) # size == (1,num_samples)
  logq_estimate = factorized_gaussian_log_density(mu_,vcat([repeat(logsig',num_samples)]...)',samples)
  return sum(logp_estimate - logq_estimate) / num_samples # average over batch
end

# Conveinence function for taking gradients
function neg_toy_elbo(params; games = two_player_toy_games(1,0), num_samples = 100)
  # TODO: Write a function that takes parameters for q,
  # evidence as an array of game outcomes,
  # and returns the -elbo estimate with num_samples many samples from q
  logp(zs) = joint_log_density(zs,games)
  return -elbo(params,logp, num_samples)
end


# Toy game
using Random
Random.seed!(1234);
num_players_toy = 2
toy_mu = randn(2) #[-2.,3.] # Initial mu, can initialize randomly!
toy_ls = randn(2) # Initual log_sigma, can initialize randomly!
toy_params_init = (toy_mu, toy_ls)

function fit_toy_variational_dist(init_params, toy_evidence; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params
  for i in 1:num_itrs
    grad_params = gradient(params -> neg_toy_elbo(params; games = toy_evidence, num_samples = num_q_samples), params_cur)[1]
    new_mu =  params_cur[1] - lr * grad_params[1]
    new_ls =  params_cur[2] - lr * grad_params[2]
    params_cur = (new_mu, new_ls)
    @info "neg_elbo: $(neg_toy_elbo(params_cur; games = toy_evidence, num_samples = num_q_samples))"
    # report the current negative elbo during training
    # TODO: plot true posterior in red and variational in blue
    # hint: call 'display' on final plot to make it display during training
    plot(title="fit_toy_variational_dist",
      xlabel = "Player 1 Skill",
      ylabel = "Player 2 Skill"
       );
    #TODO:
    target_post(zs) = exp.(joint_log_density(zs, toy_evidence)) # log
    skillcontour!(target_post,colour=:red) # plot likelihood contours for target posterior
    plot_line_equal_skill!()
    #TODO:
    mu = params_cur[1] # size = num_players
    logsig = params_cur[2]
    var_log_prior(zs) = factorized_gaussian_log_density(mu, logsig, zs)# mu_,vcat([repeat(logsig',num_q_samples)]...)',zs)
    var_post(zs) = exp.(var_log_prior(zs)) #.* exp.(all_games_log_likelihood(zs,toy_evidence))
    display(skillcontour!(var_post, colour=:blue)) # plot likelihood contours for variational posterior
  end
  return params_cur
end

#TODO: fit q with SVI observing player A winning 1 game
toy_evidence_1_0 = two_player_toy_games(1,0)
opt_params = fit_toy_variational_dist(toy_params_init, toy_evidence_1_0; num_itrs=200, lr= 1e-2, num_q_samples = 10)
num_q_samples = 10
println("neg_elbo (final loss): $(neg_toy_elbo(opt_params; games = toy_evidence_1_0, num_samples = num_q_samples))")

plot(title="Toy SVI with A:B = 1:0",
  xlabel = "Player 1 Skill",
  ylabel = "Player 2 Skill"
   )
target_post(zs) = exp.(joint_log_density(zs, toy_evidence_1_0)) # target joint, not log
skillcontour!(target_post,colour=:red) # plot likelihood contours for target posterior
plot_line_equal_skill!()

mu = opt_params[1] # size = num_players
logsig = opt_params[2]
var_log_prior(zs) = factorized_gaussian_log_density(mu, logsig, zs)# mu_,vcat([repeat(logsig',num_q_samples)]...)',zs)
var_post(zs) = exp.(var_log_prior(zs)) #.* exp.(all_games_log_likelihood(zs,toy_evidence_1_0))
display(skillcontour!(var_post, colour=:blue)) # plot likelihood contours for variational posterior
savefig(joinpath("plots","toy_svi_a1b0.png"))


#TODO: fit q with SVI observing player A winning 10 games
toy_evidence_10_0 = two_player_toy_games(10,0)
opt_params = fit_toy_variational_dist(toy_params_init, toy_evidence_10_0; num_itrs=200, lr= 1e-2, num_q_samples = 10)
num_q_samples = 10
println("neg_elbo (final loss): $(neg_toy_elbo(opt_params; games = toy_evidence_10_0, num_samples = num_q_samples))")

plot(title="Toy SVI with A:B = 10:0",
  xlabel = "Player 1 Skill",
  ylabel = "Player 2 Skill"
   )
target_post(zs) = exp.(joint_log_density(zs, toy_evidence_10_0)) # target joint, not log
skillcontour!(target_post,colour=:red) # plot likelihood contours for target posterior
plot_line_equal_skill!()

mu = opt_params[1] # size = num_players
logsig = opt_params[2]
var_log_prior(zs) = factorized_gaussian_log_density(mu, logsig, zs)
var_post(zs) = exp.(var_log_prior(zs))
display(skillcontour!(var_post, colour=:blue)) # plot likelihood contours for variational posterior
savefig(joinpath("plots","toy_svi_a10b0.png"))

#TODO: fit q with SVI observing player A winning 10 games and player B winning 10 games
toy_evidence_10_10 = two_player_toy_games(10,10)
opt_params = fit_toy_variational_dist(toy_params_init, toy_evidence_10_10; num_itrs=200, lr= 1e-2, num_q_samples = 10)
num_q_samples = 10
println("neg_elbo (final loss): $(neg_toy_elbo(opt_params; games = toy_evidence_10_10, num_samples = num_q_samples))")

plot(title="Toy SVI with A:B = 10:10",
  xlabel = "Player 1 Skill",
  ylabel = "Player 2 Skill"
   )
target_post(zs) = exp.(joint_log_density(zs, toy_evidence_10_10)) # target joint, not log
skillcontour!(target_post,colour=:red) # plot likelihood contours for target posterior
plot_line_equal_skill!()

mu = opt_params[1] # size = num_players
logsig = opt_params[2]
var_log_prior(zs) = factorized_gaussian_log_density(mu, logsig, zs)
var_post(zs) = exp.(var_log_prior(zs))
display(skillcontour!(var_post, colour=:blue))
# plot likelihood contours for variational posterior
savefig(joinpath("plots","toy_svi_a10b10.png"))

## Question 4
# Load the Data
using MAT
vars = matread("tennis_data.mat")
player_names = vars["W"]
tennis_games = Int.(vars["G"])
num_players = length(player_names)
print("Loaded data for $num_players players")


function fit_variational_dist(init_params, tennis_games; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params
  for i in 1:num_itrs
    grad_params = gradient(params -> neg_toy_elbo(params; games = tennis_games, num_samples = num_q_samples), params_cur)[1]
    new_mu =  params_cur[1] - lr * grad_params[1]
    new_ls =  params_cur[2] - lr * grad_params[2]
    params_cur = (new_mu, new_ls)
    @info "neg_elbo: $(neg_toy_elbo(params_cur; games = tennis_games, num_samples = num_q_samples))"
  end
  println("neg_elbo (final loss): $(neg_toy_elbo(params_cur; games = tennis_games, num_samples = num_q_samples))")
  return params_cur
end

using Random
Random.seed!(1234);
# TODO: Initialize variational family
init_mu = randn(num_players)
init_log_sigma = randn(num_players)
init_params = (init_mu, init_log_sigma)

# Train variational distribution
trained_params = fit_variational_dist(init_params, tennis_games)
means = trained_params[1][:]
logstd = trained_params[2][:]

perm = sortperm(means)
plot(title="Mean and variance of all players",
  xlabel = "Players sorted by skills",
  ylabel = "Approximate Player Skill"
   )
plot!(means[perm], yerror=exp.(logstd[perm]), label="Skill")
savefig(joinpath("plots","player_mean_var.png"))

#TODO: 10 players with highest mean skill under variational model
#hint: use sortperm
desc_perm = sortperm(means, rev=true)
println("Top 10 players: $(player_names[desc_perm][1:10])")

#TODO: joint posterior over "Roger-Federer" and ""Rafael-Nadal""
#hint: findall function to find the index of these players in player_names
RF_idx = findall(x->x=="Roger-Federer", player_names)
RN_idx = findall(x->x=="Rafael-Nadal", player_names)

plot(title="Roger Federer vs. Rafael Nadal",
  xlabel = "Roger Federer's Skill",
  ylabel = "Rafael Nadal's Skill"
   )
plot!(range(-3, 3, length=200), range(-3, 3, length=200), label="Equal Skill", legend=:topleft)
mu = [reshape(means[RF_idx],1)[1], reshape(means[RN_idx],1)[1]]
logsig = [reshape(logstd[RF_idx],1)[1], reshape(logstd[RN_idx],1)[1]]
var_log_prior(zs) = factorized_gaussian_log_density(mu, logsig, zs)
var_post(zs) = exp.(var_log_prior(zs))
display(skillcontour!(var_post))
savefig(joinpath("plots","RF_RN.png"))

using Distributions
mu_RF = 2.3921797157158498
mu_RN = 2.3064684607373023
var_RF = exp(-1.1412578223971224)^2
var_RN = exp(-1.2163985016732244)^2
# (g) exact prob
exact_g = 1 - cdf(Normal(0,1), (mu_RN - mu_RF)/sqrt(var_RF + var_RN))

lowest_idx = desc_perm[num_players]
mu_lowest = means[lowest_idx]
var_lowest = exp(logstd[lowest_idx])^2
# (h) exact prob
exact_h = 1 - cdf(Normal(0,1), (mu_lowest - mu_RF)/sqrt(var_RF + var_lowest))

# (g) Simple MC
using Random
Random.seed!(1234);
MC_size = 10000
samples_RF = randn(MC_size) * exp(-1.1412578223971224) .+ mu_RF
samples_RN = randn(MC_size) * exp(-1.2163985016732244) .+ mu_RN
MC_g = count(x->x==1,samples_RF .> samples_RN) / MC_size

# (h) Simple MC
samples_lowest = randn(MC_size) * exp(logstd[lowest_idx]) .+ mu_lowest
MC_h = count(x->x==1,samples_RF .> samples_lowest) / MC_size
