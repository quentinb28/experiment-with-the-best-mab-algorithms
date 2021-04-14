<h1 align="center">
  Experiment With Multi-Armed Bandit Algorithms

</p>

<img src="https://github.com/quentinb28/experiment-with-mab-algorithms/blob/main/images/experiment-with-mab-algorithms.png" width=100%>

<p align="center">
 <img src="https://img.shields.io/badge/sql-v2017-pink.svg" />
 <img src="https://img.shields.io/badge/python-v3.7-yellow.svg" />
 <img src="https://img.shields.io/badge/pandas-v1.2-red.svg" />
 <img src="https://img.shields.io/badge/plotly-v4.14-purple.svg" />
 <img src="https://img.shields.io/badge/dash-v1.19-green.svg" />
 <img src="https://img.shields.io/badge/docker_image-v1-informational.svg" />
</p>

# Table of Contents

[Background](#background)

[Part 1: Greedy Algorithms]

[Part 2: Optimistic Initial Values]

[Part 3: UCB1]

[Part 4: Thompson Sampling]

[Non Stationary Environment]

[Conclusion]

# Background
A business is about the continuous improvement of its products, services and processes. If they are to survive in today's fast-paced environment, businesses have to adopt a lean mentality-fail fast and learn quickly.
Online advertising is seeing many actors compete in apt ways for our attention. Ads get shown if they generate the best profit. It follows that companies employ many techniques and performance measures to make quick decisions about which Ads are worth delivering.
Imagine we are an advertising agency, and our clients pay us for every click they get on their Ad that was delivered on our page. The success rate in that case could be the Click-Through Rate (or CTR) - number of clicks divided by number of impressions. Our client gives us three options to choose from. Logically, we would like to send all our traffic to the Ad with the highest CTR, as it is the most profitable. In reality many additional parameters may come into play to make one such decision- Client Budget, Site Coverage, Click Prices to name just a few but I will solely focus on the CTR for now.
As we do not know in advance the CTR for each option, we would need to test and send traffic to an unlimited number of samples to reach a statistically compelling conclusion. Indeed, as we collect more samples, the confidence interval of our estimate decreases and we become more confident about the Ad most profitable for us. 
Do you see the problem here? 
Yes! We would have to "waste" part of our traffic for less profitable Ads…
This trade-off is referred to as the Exploration Exploitation Dilemma.
The very good news is that many brilliant people have worked on this topic for years and there exist many solutions to tackle this issue. My objective here is simply to lay out an overview of the main solutions and how they perform. 
The content is largely based on the course Bayesian Machine Learning in Python: A/B Testing by the Lazy Programmer that you can find on Udemy. I strongly recommend you take this course if you have any interest in A/B testing and Bayesian approaches!

Part 1: Greedy Algorithms
Let's take our Online Advertising perspective. A naïve and well known approach is called Greedy as it would always pick the Ad with the highest CTR irrespective of the confidence in the prediction or the amount of data collected. In other words, we exploit a hundred percent of the times and leave no room for exploration except for the first iteration.

CUMUL PERF SCREENSHOT

What you have here is the cumulative performance of the Greedy algorithm after 1000 iterations and for three options A, B and C with success rates of 0.25, 0.50 and 0.75 respectively. This approach fails to identify the best Ad as we progress in the experiment and this results in a poor cumulated return. Our performance, which corresponds to our Cumulated Gain divided by our Total Expected Gain, averages 33%.

Epsilon Greedy (Ɛ)
One alternative to this naïve approach is called Epsilon Greedy. This approach allows for some exploration that can help us improve our current knowledge about each Ad. For example, imagine we have Ɛ = 10%, this means that in ten percent of the cases we select randomly amongst all the options available and update their respective estimate and in ninety percent of the cases we exploit based on the estimates that we know - picking the highest CTR at iteration i.

CUMUL PERF SCREENSHOT

Although the parameters of the experiment remain the same, we can clearly see that the exploration threshold (Ɛ) allows us to make better predictions as of which Ad to select for each iteration. Our performance is 94% here, much better than our previous 33%.
However, this algorithm will never stop exploring irrespective of our confidence about the success rates and we are limiting the expected gains as a result. In other words, we will still select an Ad randomly in ten percent of the cases if Ɛ = 10% even if we already know with some high confidence which is the most profitable Ad. The total expected gain in our case follows:
Total Expected Reward = (1 - Ɛ) * 0.75 + Ɛ * ((0.25 + 0.50 + 0.75) /3)
The above expression means that, by design, we are limiting the maximum reward that we can expect - that is, the highest CTR among all Ads (0.75).
Having said that, this can prove useful in non-stationary situations. More on that later.

Part 2: Optimistic Initial Values
In the greedy algorithms we are picking each time the Ad having the best estimate at iteration i and update its CTR with the new outcome - click or not click. Here we are doing the same except that we do not start at zero but rather at some predetermined value that we call an Optimistic Initial Value.

CUMUL PERF SCREENSHOT

We compute the estimate based on this initial value. This allows for some exploration in the early stages until the best option settles at its true mean while the other options keep deteriorating until they settle below the best CTR threshold. In other words, at some stage and with sufficient iterations, only the best Ad will be selected.

Part 3: UCB1
The Upper Confidence Bound (or ucb1) algorithm aims to convert a set of average rewards at iteration i into a set of decision values takin into account how many times each Ad has been delivered so far. In other words, for each iteration we want to select the best Ad based on some decision value (or DV) that is equal to their current estimate plus some upper bound.

DV = Ad Estimate+ square root (2 * log(Total Plays) / Total Ad Plays)

CUMUL PERF SCREENSHOT

Intuitively, we understand that as we move along the process, if the number of times a certain Ad is delivered does not increase (Total Ad Plays), then its decision value will increase and with it, its chances of being selected next.
This is because the number of total iterations (Total Plays) is in the numerator and the number of times the Ad is delivered is in the denominator.
Don't be too distracted by the math here. At the end of the day, in practice, you will choose your hyper-parameter (or heuristic) and update the formula until you get a satisfactory result. It is trial and error. Have fun!

Visually, you can see that although a3 (green) seems to perform better with a current estimate at around 2.5 units, it might be worth picking a1 (blue) or a2 (red) for the next iteration as, in comparison, their confidence interval is larger, and we are less certain about their true success rate as a result. This gives a chance to other options that potentially have not been delivered enough times to make a conclusive decision about their performance.

Part 4: Thompson Sampling
Since my objective is not to enter into too many technical details, I'll pass on the complicated math and focus on the intuition behind it. If you are interested, check out Bayes' Theorem!
Thompson Sampling is what we call a Bayesian approach (or BA) as opposed to a Frequentist approach (or FA). In simple terms, the former maintains a probability distribution that represents the uncertainty about the parameters whereas the latter considers the parameters to be fixed. For example, in our online advertising example, say, we have iterated ten times and in each case we obtained either a click (C) or merely a view without a click (NC).
[C, NC, C, C, NC, C, C, NC, NC, NC]
The FA would guess at the end of the experiment that the probability of getting a click next is 50% - 5 x C/ 10 iterations. On the other end, the BA would consider the success rate - chances of getting a click- to be a random variable and will maintain a probability distribution that represents the uncertainty about the CTR.
Thompson Sampling comes in different flavors and can be used to model distributions such as Bernoulli, Beta, Gaussian, Poisson and others. It looks like the Beta distribution is the most suitable for us here as it is a continuous distribution within support of (0, 1). Other scenarios might call for other kind of distributions. This means that an Ad is delivered and the outcome affects the distribution of the uncertainty around the chances of getting a click next.

At first (yellow) we start with a uniform distribution of the probabilities of getting a click. This initial distribution is called a prior - we could choose whatever initial prior we see best fit depending on our own experience. As we deliver more Ads, the distribution is being updated and changes shape. The new, updated distribution is called a posterior which will be a prior for the next iteration and so on and so forth. After a few iterations, we got two clicks and four not clicks (green) and as you can see the distribution adjusted and we are now more confident about the fact that we are less likely to get a click next as opposed to the other scenarios (blue and purple). A FA would say that the probability of getting a click is 1/3, which is the mean of our distribution, but here we have the full overview of the uncertainty around that CTR.

Three hundred iterations and we already reached a performance of close to 98%. Thompson Sampling is fast at identifying the best option and remains flexible as it takes into account the full distributions of the success rates for the other Ads.

Non Stationary Environment
Although the performances here are measured in a stationary environment - the success rates or CTR do not change over time - in real life, namely in online scenarios, things may get more complex and the success rates may change over time. A mean computed from "old" measures will be outdated, if not detrimental to support us in making the right decisions.
Algorithms such as Thompson Sampling or Epsilon Greedy account for that as they leave some room for exploration of the other options and thereby a chance to readjust their success rate estimates with the new data collected.
Another approach would be to compute an Exponential Weighed Moving Average (EWMA) at each iteration or in other words give more weigh to more recent measures than to older ones, less relevant to our current situation.

Conclusion
I encourage you to pull the little Dash app I created and play around with it to see how the algorithms react. Below is a table of the algorithms performances that you can find within the app. You will be able to enter your own parameters and try it out. That specific table below is based on some one thousand iterations and success rates of 0.25, 0.50 and 0.75 for SlotA, SlotB and SlotC respectively. It averages the metrics over one hundred repetitions.

After many trials on my part, I see Thompson Sampling and Optimistic Initial Values always at the top regardless of how low, how high and how close the success rates are or how many iterations I consider.
However, the Optimistic Initial Values might not be as well suited for more complex, ever-changing problems that we may face on our day-to-day as Thompson Sampling for the aforementioned reasons. Each case should have its tailor-made solution that awaits its designer, YOU!
