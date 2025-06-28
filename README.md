# Credit Scoring Business Understanding

## 1.How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord emphasizes the importance of accurate, transparent, and risk-sensitive credit risk measurement. In the context of developing a credit scoring model, this regulation requires financial institutions like Bati Bank to adopt Internal Ratings-Based (IRB) approaches only if they can demonstrate that their models are sound, reliable, and interpretable.

Therefore, we must:
* **Document every modeling decision** clearly.
* **Ensure traceability and auditability** of predictions.
* **Use interpretable models** that can be explained to regulators, auditors, and other stakeholders.

This regulatory requirement motivates the adoption of models like Logistic Regression with Weight of Evidence (WoE) transformations which provide clear reasoning behind credit decisions.

## 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In the absence of a direct default label in our dataset, we must create a **proxy target** to identify high-risk customers. This is achieved by using **RFM (Recency, Frequency, Monetary)** analysis and unsupervised clustering to label customers based on their engagement and purchasing behavior.

**Why necessary:**

* It enables the training of supervised models when true default information is missing.
* Allows faster prototyping and iteration based on available data.

**Risks:**

* The proxy may not perfectly reflect actual default behavior.
* Model may misclassify genuinely good customers as high-risk (false positives), impacting customer satisfaction.
* Lending to misclassified low-risk customers who are actually high-risk (false negatives) can lead to financial loss.

Hence, ongoing model monitoring and eventual validation with real default data is critical to mitigate these risks.

## 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

## Models with their Pros and Cons
### Simple & Interpretable (Logistic Regression + WoE)
     Pros: Easy to explain- Regulatory compliant- Stable & low variance
     Cons: May underperform on complex patterns- Less flexible

### Complex & High-Performance (GBM, Random Forest)
     Pros: High predictive accuracy- Captures nonlinear relationships- Can improve business KPIs
     Cons: Difficult to interpret- Harder to document- Regulatory hurdles

Trade-off Summary: In regulated contexts like banking, simplicity and transparency often outweigh marginal performance gains. Therefore, a hybrid approach may be used: start with interpretable models for deployment and compliance, while using complex models in the background for decision support or risk flagging (subject to validation).