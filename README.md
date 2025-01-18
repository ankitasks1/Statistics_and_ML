# Statistics_and_ML
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
