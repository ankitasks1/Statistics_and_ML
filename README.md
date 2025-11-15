# Statistics_and_ML

````markdown
# 🧠 Comprehensive Guide to Statistical Tests and Distributions in R

> **Author:** Ankit Verma
> **Audience:**  Data scientists  
> **Language:** R  
> **Purpose:** Understand, apply, and visualize major parametric and non-parametric tests, hypothesis testing, and probability distributions.
````
---

## 📘 Table of Contents
1. [Parametric vs Non-Parametric Tests](#parametric-vs-non-parametric-tests)
2. [Hypothesis Testing](#hypothesis-testing)
3. [t-Test](#t-test)
4. [z-Test](#z-test)
5. [Kolmogorov–Smirnov Test](#kolmogorov–smirnov-test)
6. [Hypergeometric Test](#hypergeometric-test)
7. [Shapiro–Wilk Test](#shapiro-wilk-test)
8. [Wilcoxon Rank-Sum Test](#wilcoxon-rank-sum-test)
9. [Wilcoxon Signed-Rank Test](#wilcoxon-signed-rank-test)
10. [Kruskal–Wallis Test](#kruskal-wallis-test)
11. [ANOVA](#anova)
12. [Chi-Square Test](#chi-square-test)
13. [Correlation Tests (Pearson & Spearman)](#correlation-tests)
14. [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
15. [Bayesian Model & Posterior Probability](#bayesian-model--posterior-probability)
16. [Cox Proportional Hazards Model](#cox-proportional-hazards-model)
17. [Kaplan–Meier Survival Analysis](#kaplan-meier-survival-analysis)
18. [Common Distributions in R](#common-distributions-in-r)
19. [Summary Table](#summary-table)
20. [Data Distributions Type](#data-distributions)

---
## Power of a test
The power of a hypothesis test is the probability of rejecting the null hypothesis when the alternative hypothesis is the hypothesis that is true


## 📊 Parametric vs Non-Parametric Tests

| Type | Description | Example | R Function | Assumptions |
|------|--------------|----------|-------------|--------------|
| **Parametric** | Assumes a specific distribution (usually normal) | t-test, ANOVA, Pearson correlation | `t.test()`, `aov()`, `cor.test(..., "pearson")` | Normality, equal variance |
| **Non-Parametric** | Distribution-free; uses ranks | Wilcoxon, Kruskal–Wallis, Spearman | `wilcox.test()`, `kruskal.test()`, `cor.test(..., "spearman")` | Fewer assumptions |

**Example in R:**
```r
x <- rnorm(20, mean=5)
y <- rnorm(20, mean=6)

# Parametric
t.test(x, y)

# Non-parametric
wilcox.test(x, y)
````

---

## 🧪 Hypothesis Testing

**Steps:**

1. State Null (H₀) and Alternative (H₁)
2. Choose significance level (α = 0.05)
3. Compute test statistic
4. Obtain p-value
5. Decide: reject or fail to reject H₀

---

## 🎯 t-Test

**Purpose:** Compare means between two groups (σ unknown, small n)

```r
x <- rnorm(15, 5)
y <- rnorm(15, 6)
t.test(x, y, var.equal = TRUE)
```

**Assumptions:**

* Normality
* Equal variances
* Independence

---

## 🧮 z-Test

**Purpose:** Compare means (large n or known σ)

```r
x <- rnorm(40, 100, 15)
z_value <- (mean(x) - 100) / (15/sqrt(length(x)))
p_value <- 2 * (1 - pnorm(abs(z_value)))
p_value
```

---

## 🌊 Kolmogorov–Smirnov Test

**Purpose:** Compare distribution of sample vs. theoretical distribution

#### Null hypthesis of KS-test
H₀: The two distributions are identical.
For one sample KS-test we assess for particular distribution patterns and for two sample KS-test we assess if two given continuous distributions are identical.


```r
x <- rnorm(100)
ks.test(x, "pnorm", mean=0, sd=1)
```

**Assumption:** Continuous data, no ties

---

## 🎯 Hypergeometric Test

**Used in:** Gene set enrichment or sampling without replacement

```r
phyper(q=3, m=10, n=90, k=10, lower.tail=FALSE)
```

---

## 📏 Shapiro–Wilk Test

**Purpose:** Test for normality

```r
x <- rnorm(50)
shapiro.test(x)
```

---

## 🧾 Wilcoxon Rank-Sum Test (Mann–Whitney U)

**Purpose:** Compare two independent groups (non-parametric t-test)

```r
x <- rnorm(10, 5)
y <- rnorm(10, 6)
wilcox.test(x, y)
```

Concept: 
The Wilcoxon rank-sum test (also called the Mann–Whitney U test) is a nonparametric alternative to the two-sample t-test.
It tests whether two independent groups come from populations with the same median (or central location) — without assuming a normal distribution.

So instead of comparing means, the test compares the ranks of all observations combined.

Steps:

1. Combine the two groups into one list.

2. Rank all values from smallest to largest (1 = smallest, N = largest).

3. Compute the sum of ranks for each group.

4. If one group consistently has higher ranks, it indicates a shift in central tendency (median difference).


Example:

A <- c(2.1,2.3)

B <- c(3.4, 3,6, 1.2)

Combine A and B values and rank them: 

1:{1.2} | 2:{2.1} | 3:{2.3}  | 4:{3.4}  | 5:{3.6} 

Now sum the ranks separately for different groups: A= 2+3=5, B=1+4+5=10

If one group has consistently higher ranks, it likely has a higher median.

---

## 🧮 Wilcoxon Signed-Rank Test

**Purpose:** Paired sample comparison (non-parametric paired t-test)

```r
x <- rnorm(10, 5)
y <- x + rnorm(10, 0.5)
wilcox.test(x, y, paired=TRUE)
```

---

## 📊 Kruskal–Wallis Test

**Purpose:** Compare ≥3 independent groups (non-parametric ANOVA)

```r
data <- data.frame(
  value = c(rnorm(10,5), rnorm(10,6), rnorm(10,7)),
  group = factor(rep(1:3, each=10))
)
kruskal.test(value ~ group, data)
```

---

## 📈 ANOVA

**Purpose:** Compare means among ≥3 groups

```r
data <- data.frame(
  value = c(rnorm(10,5), rnorm(10,6), rnorm(10,7)),
  group = factor(rep(1:3, each=10))
)
res <- aov(value ~ group, data)
summary(res)
```

---

## 🧮 Chi-Square Test

**Purpose:** Test association between categorical variables

```r
tbl <- matrix(c(10, 20, 20, 50), nrow=2)
chisq.test(tbl)
```
Checking if survival is dependent of gene X mutation

Total Patient = 236

Patient with mutation in gene X = 82

Patient without mutation in gene X = 236 - 82 = 154

Died within one Year of taking drug =  87

Survive within one Year of taking drug =  236 - 87 = 149

Total survivors =  149

-->Survivors with gene X mutation = 42

-->Survivors without gene X mutation = 149 - 42 = 107

Total died = 87

-->Died with gene X mutation = 82-42 = 40

-->Died without gene X mutation = 87 - 40 = 47

Create contingency table, 

Survivor | Died

cont_tab <- matrix(c(42,107,40,47), nrow = 2)

rownames(cont_tab) <- c("Mutation", "Normal")

colnames(cont_tab) <- c("Survived", "Died")

chisq.test(cont_tab)

---

## 🔗 Correlation Tests

**Pearson (parametric):**

```r
x <- rnorm(30)
y <- x + rnorm(30)
cor.test(x, y, method="pearson")
```

**Spearman (non-parametric):**

```r
cor.test(x, y, method="spearman")
```

---

## 📉 Maximum Likelihood Estimation

**Example:** Estimate parameters of normal distribution

```r
x <- rnorm(100, mean=5, sd=2)
mle_mean <- mean(x)
mle_sd <- sqrt(mean((x - mle_mean)^2))
mle_mean; mle_sd
```

---

## 🧠 Bayesian Model & Posterior Probability

**Formula:** Posterior = (Likelihood × Prior) / Evidence

```r
prior <- 0.5
likelihood <- 0.9
evidence <- (0.9*0.5 + 0.1*0.5)
posterior <- (likelihood * prior) / evidence
posterior
```

---

## ⏱️ Cox Proportional Hazards Model

**Purpose:** Examine survival time and covariates

```r
library(survival)
data(lung)
coxph(Surv(time, status) ~ age + sex, data=lung)
```

---

## 📆 Kaplan–Meier Survival Analysis

**Purpose:** Estimate survival probabilities

```r
library(survival)
data(lung)
fit <- survfit(Surv(time, status) ~ sex, data=lung)
plot(fit)
```

---

## 📊 Common Distributions in R

| Distribution          | Description               | Example R Code                   |
| --------------------- | ------------------------- | -------------------------------- |
| **Normal**            | Continuous symmetric data | `rnorm(100, mean=0, sd=1)`       |
| **Binomial**          | #successes in n trials    | `rbinom(100, size=10, prob=0.5)` |
| **Poisson**           | Counts (rare events)      | `rpois(100, lambda=3)`           |
| **Negative Binomial** | Overdispersed counts      | `rnbinom(100, mu=10, size=2)`    |
| **Exponential**       | Time between events       | `rexp(100, rate=1)`              |
| **Uniform**           | Equal probability         | `runif(100, min=0, max=1)`       |

---

## 🧭 Summary Table

| Test           | Type           | Purpose                  | R Function                  |
| -------------- | -------------- | ------------------------ | --------------------------- |
| t-Test         | Parametric     | Compare means (2 groups) | `t.test()`                  |
| z-Test         | Parametric     | Compare means (known σ)  | Manual                      |
| ANOVA          | Parametric     | Compare ≥3 means         | `aov()`                     |
| Chi-Square     | Non-parametric | Categorical association  | `chisq.test()`              |
| KS-Test        | Non-parametric | Compare distributions    | `ks.test()`                 |
| Wilcoxon       | Non-parametric | Median difference        | `wilcox.test()`             |
| Kruskal–Wallis | Non-parametric | ≥3 groups                | `kruskal.test()`            |
| Shapiro–Wilk   | -              | Normality                | `shapiro.test()`            |
| Pearson        | Parametric     | Linear correlation       | `cor.test(..., "pearson")`  |
| Spearman       | Non-parametric | Rank correlation         | `cor.test(..., "spearman")` |
| Cox            | Parametric     | Survival                 | `coxph()`                   |
| Kaplan–Meier   | Non-parametric | Survival probability     | `survfit()`                 |

---

## 📚 References

* R Documentation: [https://www.rdocumentation.org/](https://www.rdocumentation.org/)
* Hastie, Tibshirani & Friedman (2009). *The Elements of Statistical Learning*
* OpenIntro Statistics: [https://www.openintro.org/](https://www.openintro.org/)
* R for Data Science: [https://r4ds.had.co.nz/](https://r4ds.had.co.nz/)

---

## Data Distributions Type
‐----‐-----------------------------
Below is a summary of the probability density function (PDF) formulas for some of the most commonly used probability distributions. These distributions can describe continuous random variables, and each has a different PDF that corresponds to a specific type of behavior or data.

---‐-------------------------------

### 

````markdown
# 🎲 Probability Distributions in R

---

## 1. 🟢 Normal Distribution (Gaussian Distribution)

**Description:**  
The **Normal Distribution** is symmetric and bell-shaped, commonly used to model natural phenomena like height, measurement error, etc.

**PDF:**
\[
f(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
\]

Where:  
- \( \mu \): Mean (center of distribution)  
- \( \sigma^2 \): Variance (spread)

**R Example:**
```r
set.seed(123)
x <- rnorm(1000, mean=0, sd=1)
hist(x, probability=TRUE, col="skyblue", main="Normal Distribution")
lines(density(x), col="red", lwd=2)
````

---

## 2. 🟠 Exponential Distribution

**Description:**
Models **time between events** in a Poisson process (e.g., time between arrivals).

**PDF:**
[
f(x|\lambda) = \lambda e^{-\lambda x}, \quad x \geq 0
]

Where:

* ( \lambda ): Rate parameter (mean = ( 1/\lambda ))

**R Example:**

```r
x <- rexp(1000, rate=1)
hist(x, probability=TRUE, col="lightgreen", main="Exponential Distribution")
curve(dexp(x, rate=1), add=TRUE, col="red", lwd=2)
```

---

## 3. 🟣 Uniform Distribution

**Description:**
All values are **equally likely** within the interval [a, b].

**PDF:**
[
f(x|a,b) = \frac{1}{b - a}, \quad a \leq x \leq b
]

**R Example:**

```r
x <- runif(1000, min=0, max=10)
hist(x, probability=TRUE, col="khaki", main="Uniform Distribution")
curve(dunif(x, min=0, max=10), add=TRUE, col="red", lwd=2)
```

---

## 4. 🔵 Gamma Distribution

**Description:**
Models **waiting times** for multiple events in a Poisson process.

**PDF:**
[
f(x|\alpha, \beta) = \frac{x^{\alpha - 1} e^{-x/\beta}}{\Gamma(\alpha)\beta^\alpha}, \quad x \geq 0
]

Where:

* ( \alpha ): Shape
* ( \beta ): Scale

**R Example:**

```r
x <- rgamma(1000, shape=2, scale=2)
hist(x, probability=TRUE, col="lightblue", main="Gamma Distribution")
curve(dgamma(x, shape=2, scale=2), add=TRUE, col="red", lwd=2)
```

---

## 5. 🔴 Beta Distribution

**Description:**
Used for random variables constrained between 0 and 1 (e.g., proportions, probabilities).

**PDF:**
[
f(x|\alpha, \beta) = \frac{x^{\alpha - 1}(1 - x)^{\beta - 1}}{B(\alpha, \beta)}, \quad 0 \leq x \leq 1
]

**R Example:**

```r
x <- rbeta(1000, shape1=2, shape2=5)
hist(x, probability=TRUE, col="plum", main="Beta Distribution")
curve(dbeta(x, shape1=2, shape2=5), add=TRUE, col="red", lwd=2)
```

---

## 6. 🟡 Log-Normal Distribution

**Description:**
If ( Y = \ln(X) ) is normally distributed, then ( X ) follows a log-normal distribution.
Used in modeling **income**, **stock prices**, etc.

**PDF:**
[
f(x|\mu, \sigma^2) = \frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(\ln(x) - \mu)^2}{2\sigma^2}}, \quad x > 0
]

**R Example:**

```r
x <- rlnorm(1000, meanlog=0, sdlog=1)
hist(x, probability=TRUE, col="orange", main="Log-Normal Distribution")
curve(dlnorm(x, meanlog=0, sdlog=1), add=TRUE, col="red", lwd=2)
```

---

## 7. ⚙️ Weibull Distribution

**Description:**
Used in **reliability analysis** and **survival studies**.
Can model increasing or decreasing failure rates.

**PDF:**
[
f(x|\lambda, k) = \frac{k}{\lambda} \left( \frac{x}{\lambda} \right)^{k-1} e^{-(x/\lambda)^k}, \quad x \geq 0
]

**R Example:**

```r
x <- rweibull(1000, shape=2, scale=1)
hist(x, probability=TRUE, col="cyan", main="Weibull Distribution")
curve(dweibull(x, shape=2, scale=1), add=TRUE, col="red", lwd=2)
```

---

## 8. 🧮 Chi-Square Distribution

**Description:**
Used in **goodness-of-fit** and **independence tests**.

**PDF:**
[
f(x|k) = \frac{x^{(k/2) - 1} e^{-x/2}}{2^{k/2} \Gamma(k/2)}, \quad x \geq 0
]

Where ( k ) = degrees of freedom.

**R Example:**

```r
x <- rchisq(1000, df=4)
hist(x, probability=TRUE, col="pink", main="Chi-Square Distribution")
curve(dchisq(x, df=4), add=TRUE, col="red", lwd=2)
```

---

## 9. 📉 Student’s t-Distribution

**Description:**
Used for **small-sample inference** when variance is unknown.

**PDF:**
[
f(x|\nu) = \frac{\Gamma\left( \frac{\nu+1}{2} \right)}{\sqrt{\nu\pi}\Gamma\left( \frac{\nu}{2} \right)} \left( 1 + \frac{x^2}{\nu} \right)^{-\frac{\nu+1}{2}}
]

**R Example:**

```r
x <- rt(1000, df=10)
hist(x, probability=TRUE, col="lightgray", main="Student's t-Distribution")
curve(dt(x, df=10), add=TRUE, col="red", lwd=2)
```

---

## 10. 💰 Pareto Distribution

**Description:**
Models **wealth/income distributions** — a few people hold most of the wealth.

**PDF:**
[
f(x|\alpha, x_m) = \frac{\alpha x_m^\alpha}{x^{\alpha+1}}, \quad x \geq x_m
]

**R Example (using VGAM package):**

```r
library(VGAM)
x <- rpareto(1000, scale=1, shape=3)
hist(x, probability=TRUE, col="wheat", main="Pareto Distribution")
curve(dpareto(x, scale=1, shape=3), add=TRUE, col="red", lwd=2)
```

---

## 11. ⚫ Cauchy Distribution

**Description:**
Heavy-tailed distribution; used in physics (resonance, noise).
Has undefined mean and variance.

**PDF:**
[
f(x|x_0, \gamma) = \frac{1}{\pi \gamma \left( 1 + \left( \frac{x - x_0}{\gamma} \right)^2 \right)}
]

**R Example:**

```r
x <- rcauchy(1000, location=0, scale=1)
hist(x, probability=TRUE, col="gray", main="Cauchy Distribution")
curve(dcauchy(x, location=0, scale=1), add=TRUE, col="red", lwd=2)
```

---

## 12. 🔁 Exponential as a Special Case of Gamma Distribution

**Description:**
When ( \alpha = 1 ), the Gamma distribution becomes the **Exponential** distribution.

**PDF:**
[
f(x|\lambda) = \lambda e^{-\lambda x}, \quad x \geq 0
]

**R Example:**

```r
x <- rgamma(1000, shape=1, rate=1)
hist(x, probability=TRUE, col="lightgreen", main="Exponential (Gamma α=1)")
curve(dgamma(x, shape=1, rate=1), add=TRUE, col="red", lwd=2)
```

---

## 📊 Summary Table

| Distribution            | R Function               | Parameters      | Common Use             |
| ----------------------- | ------------------------ | --------------- | ---------------------- |
| Normal                  | `rnorm(), dnorm()`       | mean, sd        | Natural phenomena      |
| Exponential             | `rexp(), dexp()`         | rate            | Waiting time           |
| Uniform                 | `runif(), dunif()`       | min, max        | Equal chance           |
| Gamma                   | `rgamma(), dgamma()`     | shape, scale    | Waiting times          |
| Beta                    | `rbeta(), dbeta()`       | α, β            | Probabilities [0,1]    |
| Log-Normal              | `rlnorm(), dlnorm()`     | meanlog, sdlog  | Income, prices         |
| Weibull                 | `rweibull(), dweibull()` | shape, scale    | Reliability            |
| Chi-Square              | `rchisq(), dchisq()`     | df              | Tests                  |
| Student’s t             | `rt(), dt()`             | df              | Small-sample inference |
| Pareto                  | `rpareto(), dpareto()`   | shape, scale    | Wealth                 |
| Cauchy                  | `rcauchy(), dcauchy()`   | location, scale | Heavy tails            |
| Exponential (Gamma α=1) | `rexp(), dexp()`         | rate            | Time between events    |

---

## 📚 References

* R Documentation: [https://www.rdocumentation.org](https://www.rdocumentation.org)
* Casella & Berger (2002). *Statistical Inference*
* Rice (2006). *Mathematical Statistics and Data Analysis*
* OpenIntro Statistics: [https://www.openintro.org](https://www.openintro.org)

---

```


