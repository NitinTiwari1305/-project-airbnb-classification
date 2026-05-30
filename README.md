<h1 align="center">Predicting Next Booking Destinations for Airbnb Users</h1>

<p align="center">A Multi-Class Classification & Recommendation Project</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/214205680-dccc15c4-ec86-439c-b50a-b96593c4416b.png" width="450"/>
</p>

*Obs: The business problem is fictitious, although both company and data are real.*

*The in-depth Python code explanation is available in [this](https://github.com/NitinTiwari1305/-project-airbnb-classification/blob/main/airbnb.ipynb) Jupyter Notebook.*

# 1. **Airbnb and Business Problem**
<p align="justify"> Airbnb is an online marketplace for short-term homestays whose business model relies on charging a transactional commission for each successful reservation. To optimize conversion funnels and understand user intent across digital interfaces, this data science project focuses on a recommendation challenge: <b>predicting the five most likely destination countries for a newly registered USA user.</b> Airbnb provided structured demographics for over 200,000 profiles paired with clickstream web session telemetry, allowing for the generation of prioritized booking rankings for 61,000 validation users across 12 discrete multi-class target outcomes (including domestic destinations, international markets, and 'NDF' indicating a zero-booking outcome).</p>

# 2. **Data Overview**
The data is split into core user demographics and granular clickstream web sessions recording real-time internet browsing sequences. The initial feature definitions are detailed below:

<div align="center">
<table>
<tr><th><h3>Users Data</h3></th><th><h3>Sessions Telemetry</h3></th></tr>
<tr><td>

| **Feature** | **Definition** |
|----------------------|----------------|
| id                   | Unique user identifier |
| date_account_created | The timestamp of account creation | 
| timestamp_first_active| Numerical timestamp of first recorded activity |
| date_first_booking   | Calendar date of the first booking |
| gender               | Self-reported user gender |
| age                  | Calculated age of the user |
| signup_method        | Registration channel (e.g., Facebook, Google, Email) |
| signup_flow          | The tracking page index a user registered from |
| language             | International language preference settings |
| affiliate_channel    | Paid marketing macro-channel grouping |
| affiliate_provider   | Specific marketing platform node (e.g., Google, Craigslist) |
| first_affiliate_tracked| The first marketing asset the user engaged with |
| signup_app           | Technical application layer (e.g., Web, iOS, Android) |
| first_device_type    | Initial hardware class (e.g., Windows Desktop, iPhone) |
| first_browser        | Initial browser environment (e.g., Chrome, Safari) |
| country_destination  | Multi-class target variable (12 categories) |

</td><td>

| **Feature** | **Definition** |
|----------------------|----------------|
| user_id              | Primary key mapping directly to User ID |
| action               | Specific endpoint hit (e.g., show, search_results) | 
| action_type          | Abstracted behavior type (e.g., view, click) |
| action_detail        | Contextual event logging (e.g., confirm_email_link) |
| device_type          | Active hardware class during the specific action |
| secs_elapsed         | Total continuous time elapsed between sequential actions |

</td></tr>
</table>
</div>

<i>Data source gathered directly via competitive archives on Kaggle.</i>

# 3. **Business Assumptions**
<p align="justify">
- Out of the highly correlated behavioral categorical variables ('action', 'action_type', 'action_detail'), only <b>action_type</b> was retained. It compressed hundreds of sparse strings into 28 clean feature vectors, accelerating structural matrix encoding.<br>
- Missing fields inside the 'first_affiliate_tracked' vector were imputed with 'untracked' to retain transaction integrity.<br>
- Missing values within the 'age' column were handled utilizing robust regional median aggregation values.<br>
- The 'date_first_booking' column was dropped from the training matrix since this feature is fundamentally unavailable at inference time for newly registered users, preventing structural data leakage.
</p>

# 4. **Solution Plan**
## 4.1. How was the problem solved?

<p align="justify"> To predict and rank the five most likely booking destinations for validation cohorts, the following architectural steps were executed: </p>

- <b> Understanding the Business Problem</b>: Aligning the recommendation engine parameters with Airbnb's conversion mechanics and mapping the evaluation rules.

- <b> Collecting Data</b>: Querying and downloading core user matrices and clickstream data partitions.

- <b> Data Cleaning</b>: Parsing irregular text fields, running numeric type casting, treating NaN values, and evaluating outlier boundaries.

- <b> Feature Engineering</b>: Aggregating granular clickstream sequences into customer-level behavioral metrics (e.g., log-transformed session durations and action frequencies). Full derivations are detailed in the <a href="https://github.com/NitinTiwari1305/-project-airbnb-classification/blob/main/new_features.md">Feature Dictionary</a>.

- <p align="justify"> <b> Exploratory Data Analysis (EDA)</b>: Executing multi-tiered Univariate, Bivariate, and Multivariate sweeps to validate market hypotheses. Automated data structure validation reports were generated using the Pandas Profiling library and are stored in the <a href="https://github.com/NitinTiwari1305/project-airbnb-classification/blob/main/report.html">Profiling Assets Directory</a>. Top insights are detailed in <a href="#5-top-business-insights">Section 5</a>.</p>

- <b> Data Preparation</b>: Implementing mathematical <a href="https://www.atoti.io/articles/when-to-perform-a-feature-scaling/">Rescaling Tools</a> alongside robust categorical <a href="https://www.geeksforgeeks.org/feature-encoding-techniques-machine-learning/">Encoding Methods</a> to transform sparse profiles into dense numeric tensors.

- <b> Feature Selection</b>: Truncating multi-collinear and low-variance feature boundaries using an ensemble Random Forest variable importance wrapper.

- <p align="justify"> <b> Machine Learning Modeling & Validation</b>: Training seven multi-class learners across strict stratification loops, optimizing top algorithms via Bayesian Optimization with Optuna as detailed in <a href="#6-machine-learning-models">Section 6</a>.</p>

- <p align="justify"> <b> Production Deployment </b>: Compiling predictions into an operational interface and deploying a dual Flask micro-service and Streamlit frontend analytics layer as detailed in <a href="#7-model-deployment-and-results">Section 7</a>.</p>
  
## 4.2. Tools and techniques used:
- [Python 3.10.9](https://www.python.org/downloads/release/python-3109/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) and [Sklearn](https://scikit-learn.org/stable/).
- [SQL](https://www.w3schools.com/sql/) and [PostgresSQL](https://www.postgresql.org/).
- [Jupyter Notebook](https://jupyter.org/) and [VSCode](https://code.visualstudio.com/).
- [Flask Framework](https://flask.palletsprojects.com/en/2.2.x/) and [Render Cloud Services](https://render.com/).
- [Streamlit Dashboard Library](https://streamlit.io/).
- [Git](https://git-scm.com/) and [Github](https://github.com/).
- [Exploratory Data Analysis (EDA)](https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15). 
- [Techniques for Feature Selection](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/).
- [Classification & Ensemble Algorithms (Logistic Regression, Decision Tree, Random Forest, ExtraTrees, AdaBoost, XGBoost, LightGBM)](https://scikit-learn.org/stable/modules/ensemble.html).
- [Stratified Validation Routing](https://scikit-learn.org/stable/modules/cross_validation.html), [Bayesian Hyperparameter Tuning (Optuna)](https://optuna.readthedocs.io/en/stable/index.html) and [Ranking Metrics (NDCG@K)](https://www.kaggle.com/competitions/airbnb-recruiting-new-user-bookings/overview/evaluation).

# 5. **Top Business Insights**

 - ### 1st - Active users take less than 48 hours on average from their first touchpoint to formal account creation across all markets.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/214318780-33ad6f3a-3054-4c4b-90c3-c7db9d9edd85.png" alt="drawing" width="750"/>
</p>

--- 
- ### 2nd - Structural sign-up volume undergoes an aggressive velocity expansion during the Spring season.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/214318784-96eb4214-86dc-4b9e-a449-412b980a1630.png" alt="drawing" width="750"/>
</p>

--- 
- ### 3rd - Female demographic cohorts demonstrate a 15% higher propensity to execute international bookings compared to male users.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/214318787-7ea79725-8b3c-4909-b9dd-722b4d74b44e.png" alt="drawing" width="750"/>
</p>

---

# 6. **Machine Learning Models**

<p align="justify"> Seven distinct algorithmic classifiers were developed, optimized, and cross-validated to calculate recommendation vectors tracking the top 5 destination choices for newly incoming accounts. Initial baseline metrics are sorted below: </p>

<div align="center">

| **Model Variant** | **NDCG at K (K=5)** |
|:---------------------------:|:------------------:|
| **LightGBM Classifier** | **0.8496 +/- 0.0006** |
| XGBoost Classifier | 0.8482 +/- 0.0004 |
| Random Forest Classifier | 0.8451 +/- 0.0006 |
| AdaBoost Classifier | 0.8429 +/- 0.0019 |
| Extra Trees Classifier | 0.8390 +/- 0.0008 |
| Logistic Regression | 0.8377 +/- 0.0010 |
| Decision Tree Classifier | 0.7242 +/- 0.0023 |
 
</div>

<p align="justify"> <b>LightGBM</b> was selected as the operational model due to its fast gradient-based training matrix, lower memory utilization, and lean serialization weights—making it ideal for instant multi-class inference on cloud infrastructure.</p>

<p align="justify"> Hyperparameter tuning was automated utilizing Bayesian Optimization via Optuna over an insulated evaluation split, completely isolating training boundaries to eliminate risks of feature leakage. Optuna optimization successfully advanced ranking precision: </p>

<div align="center">
<table>
<tr><th>Before Optuna Search Space Tuning</th><th>Post-Optuna Final Evaluation</th></tr>
<tr><td>

| **Model** | **NDCG at K** |
|:------------------------:|:-------------:|
| LightGBM Baseline | 0.8514 |

</td><td>
 
| **Model** | **NDCG at K** |
|:------------------------:|:-------------:|
| **Optimized LightGBM** | **0.8542** | 

</td></tr>
</table>
</div>

## <i>Metrics Definition and Interpretation</i>
<p align="justify"> Because the task demands predicting and ordering a collection of target suggestions, model performance was evaluated using <b>Normalized Discounted Cumulative Gain (NDCG) at Rank K (K=5)</b>.</p>
<p align="justify"> NDCG evaluates position-sensitive relevance, bounding results between 0.0 and 1.0. An NDCG score of 1.0 defines an ideal ranking match. It evaluates not only the model's multi-class classification accuracy but its precision in sorting candidate locations from highest to lowest conversion likelihood.</p>

# 7. **Model Deployment and Results**

<p align="justify"> The production architecture was structured into three decoupled layers: </p>
 
- <p align="justify"> <b>Step 1 (Data Layer)</b>: User and clickstream history records were modeled and stored inside a relational PostgreSQL cloud engine hosted by <a href="https://neon.tech/">Neon.tech</a>. </p>
 
 - <p align="justify"> <b>Step 2 (Inference Layer)</b>: A containerized Flask application was engineered on Render Cloud. It handles database connections, processes new session arrays on the fly, triggers the serialized LightGBM tensor, and returns structured prediction vectors straight to a persistent relational database table.</p>

 - <p align="justify"> <b>Step 3 (Analytics Presentation Layer)</b>: An interactive Streamlit frontend pulls data from the relational predictions engine, displaying the prioritized rankings for all 61,000 accounts alongside visual cohorts split by age brackets, gender distributions, and global intent maps.</p>


<div align="center">
<table>
<tr><th align="center">Production Deployment Endpoints</th></tr>
<tr><td>
 
 <div align="center">

| **Streamlit User Interface Hub** | **Flask Prediction API Engine** |
|:------------------------:|:-------------:|
| [![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://airbnb.streamlit.app/) | [![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://airbnb-predict.onrender.com) | 
</div>
 
 </td></tr>
</table>
</div>

<p align="justify"> The Flask endpoint provides single-click orchestration to process incoming batch requests, ensuring scale continuity as new telemetry arrives. Project source repositories can be reviewed in the centralized code directories.</p>

# 8. **Conclusion**
In this project the main objective was accomplished:
 <p align="justify"> <b>We engineered a multi-class recommendation engine providing ordered 5-country travel forecasts for 61,000 active users, wrapped inside an interactive Streamlit UI for cross-functional business exploration.</b> The system is supported by a robust, cloud-connected Flask inference engine designed to process new real-time leads. Additionally, exploratory sweeps isolated three strategic travel insights to assist product and growth teams with personalized marketing outreach. </p>
 
# 9. **Next Steps**
<p align="justify"> Downstream performance enhancements include:
 - Constructing higher-order interaction features tracking sequential session transitions.
 - Experimenting with deep multi-class architectures, including Neural Network architectures.
 - Scaling data infrastructure into managed enterprise frameworks like AWS.
</p>

# Contact
- nitintiwari1305@gmail.com
- [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nitintiwari1305/)
