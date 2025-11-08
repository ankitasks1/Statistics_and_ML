# Statistics_and_ML

````markdown
# 🧠 Comprehensive Guide to Statistical Tests and Distributions in R

> **Author:** Ankit Verma
> **Audience:**  Data scientists  
> **Language:** R  
> **Purpose:** Understand, apply, and visualize major parametric and non-parametric tests, hypothesis testing, and probability distributions.

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

---

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

‐----‐-----------------------------
Below is a summary of the probability density function (PDF) formulas for some of the most commonly used probability distributions. These distributions can describe continuous random variables, and each has a different PDF that corresponds to a specific type of behavior or data.

---

### 1. **Normal Distribution (Gaussian Distribution)**

The **normal distribution** is one of the most widely known and used distributions. It is characterized by a symmetric bell-shaped curve.

- **PDF**:
  \[
  f(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
  \]
  Where:
  - \( \mu \) = mean (location of the peak)
  - \( \sigma^2 \) = variance (spread)

---

### 2. **Exponential Distribution**

The **exponential distribution** is used to model the time between events in a Poisson process, such as the time between arrivals in a queue.

- **PDF**:
  \[
  f(x|\lambda) = \lambda e^{-\lambda x}, \quad x \geq 0
  \]
  Where:
  - \( \lambda \) = rate parameter (mean = \( 1/\lambda \))

---

### 3. **Uniform Distribution**

The **uniform distribution** is used when all outcomes are equally likely within a given range.

- **PDF** (Continuous uniform distribution):
  \[
  f(x|a, b) = \frac{1}{b - a}, \quad a \leq x \leq b
  \]
  Where:
  - \( a \) = minimum value
  - \( b \) = maximum value

---

### 4. **Gamma Distribution**

The **gamma distribution** is often used in queuing models and to describe waiting times for multiple independent Poisson processes.

- **PDF**:
  \[
  f(x|\alpha, \beta) = \frac{x^{\alpha - 1} e^{-x/\beta}}{\Gamma(\alpha)\beta^\alpha}, \quad x \geq 0
  \]
  Where:
  - \( \alpha \) = shape parameter
  - \( \beta \) = scale parameter
  - \( \Gamma(\alpha) \) = Gamma function

---

### 5. **Beta Distribution**

The **beta distribution** is used to model variables that are constrained to a finite range, often between 0 and 1. It is widely used in Bayesian statistics.

- **PDF**:
  \[
  f(x|\alpha, \beta) = \frac{x^{\alpha - 1}(1 - x)^{\beta - 1}}{B(\alpha, \beta)}, \quad 0 \leq x \leq 1
  \]
  Where:
  - \( \alpha \) = shape parameter
  - \( \beta \) = shape parameter
  - \( B(\alpha, \beta) \) = Beta function

---

### 6. **Log-Normal Distribution**

The **log-normal distribution** is used when the logarithm of the variable is normally distributed. This distribution is often used in modeling stock prices, income distributions, etc.

- **PDF**:
  \[
  f(x|\mu, \sigma^2) = \frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(\ln(x) - \mu)^2}{2\sigma^2}}, \quad x > 0
  \]
  Where:
  - \( \mu \) = mean of the logarithm of the variable
  - \( \sigma^2 \) = variance of the logarithm of the variable

---

### 7. **Weibull Distribution**

The **Weibull distribution** is often used in reliability analysis and survival studies. It can model a variety of data behaviors, such as increasing or decreasing hazard rates.

- **PDF**:
  \[
  f(x|\lambda, k) = \frac{k}{\lambda} \left( \frac{x}{\lambda} \right)^{k-1} e^{-(x/\lambda)^k}, \quad x \geq 0
  \]
  Where:
  - \( \lambda \) = scale parameter
  - \( k \) = shape parameter

---

### 8. **Chi-Square Distribution**

The **chi-square distribution** is widely used in statistical tests, such as the goodness-of-fit test, and in confidence interval estimation.

- **PDF**:
  \[
  f(x|k) = \frac{x^{(k/2) - 1} e^{-x/2}}{2^{k/2} \Gamma(k/2)}, \quad x \geq 0
  \]
  Where:
  - \( k \) = degrees of freedom

---

### 9. **Student’s t-Distribution**

The **t-distribution** is used in hypothesis testing and confidence intervals, especially when the sample size is small.

- **PDF**:
  \[
  f(x|\nu) = \frac{\Gamma\left( \frac{\nu+1}{2} \right)}{\sqrt{\nu\pi}\Gamma\left( \frac{\nu}{2} \right)} \left( 1 + \frac{x^2}{\nu} \right)^{-\frac{\nu+1}{2}}, \quad -\infty < x < \infty
  \]
  Where:
  - \( \nu \) = degrees of freedom

---

### 10. **Pareto Distribution**

The **Pareto distribution** is often used to describe distributions of wealth or income, where a small number of individuals control a large portion of the total wealth.

- **PDF**:
  \[
  f(x|\alpha, x_m) = \frac{\alpha x_m^\alpha}{x^{\alpha+1}}, \quad x \geq x_m
  \]
  Where:
  - \( \alpha \) = shape parameter
  - \( x_m \) = minimum value of \( x \)

---

### 11. **Cauchy Distribution**

The **Cauchy distribution** has heavy tails and is often used in physics to describe resonance phenomena and other types of extreme events.

- **PDF**:
  \[
  f(x|x_0, \gamma) = \frac{1}{\pi \gamma \left( 1 + \left( \frac{x - x_0}{\gamma} \right)^2 \right)}, \quad -\infty < x < \infty
  \]
  Where:
  - \( x_0 \) = location parameter
  - \( \gamma \) = scale parameter

---

### 12. **Gamma Distribution (Special Case: Exponential Distribution)**

The **exponential distribution** is a special case of the **gamma distribution** where \( \alpha = 1 \). It is widely used to model waiting times or lifetimes.

- **PDF**:
  \[
  f(x|\lambda) = \lambda e^{-\lambda x}, \quad x \geq 0
  \]
  Where:
  - \( \lambda \) = rate parameter

---

These are just some of the well-known continuous probability distributions. Each distribution is useful for different types of data or modeling needs, and their PDFs have various forms depending on the specific characteristics of the data they represent.
