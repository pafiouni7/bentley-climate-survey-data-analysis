-----------------------------
Packages
-----------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

--------------------------
Import Data
--------------------------

#mount google drive
from google.colab import drive
drive.mount('/content/drive')

NOTEBOOK_PATH =  "/content/drive/MyDrive/Uncompetition/"

df = pd.read_excel(NOTEBOOK_PATH + "/bex_data_2024_public.xlsx")

----------------------------
Initial Data Analysis
----------------------------

df.shape
df.info()

----------------------------------------------------
Bentley Institutional Values and Demographics Data
----------------------------------------------------

values_df = df.iloc[:, list(range(4, 8)) + list(range(10, 12))+ list(range(145, 152))]

values_df.rename(columns={'bentley_group': 'group',
                          'demo_urm': 'race',
                          'demo_sex': 'gender',
                          'demo_ses': 'ses',
                          'stu_diverse_grewup': 'diversity_gu',
                          'stu_diverse_school': 'diversity_hs',
                          'bentley_values_01': 'caring',
                          'bentley_values_02': 'collaboration',
                          'bentley_values_03': 'diversity',
                          'bentley_values_04': 'honesty',
                          'bentley_values_05': 'impact',
                          'bentley_values_06': 'learning',
                          'bentley_values_07': 'respect'}, inplace=True)

values_df = values_df[(values_df['race'] != "Not Specified") &
                      (values_df['gender'] != "Not Specified or Undisclosed Intersex")]

values_df.head()

--------------------------
Exploratory Data Analysis
--------------------------

values_df.caring.value_counts(dropna = False)
values_df.collaboration.value_counts(dropna = False)
values_df.diversity.value_counts(dropna = False)
values_df.honesty.value_counts(dropna = False)
values_df.impact.value_counts(dropna = False)
values_df.learning.value_counts(dropna = False)
values_df.respect.value_counts(dropna = False)

values = ['caring', 'collaboration', 'diversity', 'honesty', 'impact', 'learning', 'respect']

values_df[values].mean()

#diversity lowest at ~3.80
#learning highest at ~4.21

group_df = values_df.groupby('group')[values].mean()
group_df

group_df['group_sum'] = values_df.groupby('group')[values].mean().sum(axis=1)
group_df

#Graduates tend to give less harsh evaluations of the core values

group_means = values_df.groupby('group')[values].mean()
group_means.max() - group_means.min()

#Collaboration shows the largest difference in perception across groups, indicating the strongest disagreement between students, staff, and other populations.

-------------------------------------------------------------------
Visualizations (Matrices, Boxplots, Heatmaps)
-------------------------------------------------------------------

sns.heatmap(group_means, annot=True, cmap='Blues')
plt.title('Core Values by Group')
plt.show()

#Faculty tends to give harsh evaluations of the core values.
#Learning is the leading core value across highest evaluations.
#Diversity is the leading core value across lowest evaluations.

for value in values:
    sns.boxplot(x='group', y=value, data=values_df)
    plt.title(f'{value.capitalize()} by Group')
    plt.show()

sns.heatmap(values_df[values].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Core Values')
plt.show()

#People who rate one core value highly tend to rate all others highly too.
#Values like caring, honesty, and respect are closely connected, suggesting that individuals who perceive the university as strong in one of these areas tend to view it positively across the others.

---------------------------------
Regression Analysis
---------------------------------

model1 = sm.ols('caring ~ C(group) + C(race) + C(gender) + ses', data = values_df).fit()
print(model1.summary())
model2 = sm.ols('collaboration ~ C(group) + C(race) + C(gender) + ses', data = values_df).fit()
print(model2.summary())
model3 = sm.ols('diversity ~ C(group) + C(race) + C(gender) + ses', data = values_df).fit()
print(model3.summary())
model4 = sm.ols('honesty ~ C(group) + C(race) + C(gender) + ses', data = values_df).fit()
print(model4.summary())
model5 = sm.ols('impact ~ C(group) + C(race) + C(gender) + ses', data = values_df).fit()
print(model5.summary())
model6 = sm.ols('learning ~ C(group) + C(race) + C(gender) + ses', data = values_df).fit()
print(model6.summary())
model7 = sm.ols('respect ~ C(group) + C(race) + C(gender) + ses', data = values_df).fit()
print(model7.summary())

#Regression results show that group membership and socioeconomic status are significant predictors of how individuals rate the university’s core values.
#Gender and race are not statistically significant predictors, indicating no meaningful difference in core value ratings between males and females or between URM and non-URM respondents.

-----------------
Classification
-----------------

values_df['overall_score'] = values_df[['caring','collaboration','diversity','honesty','impact','learning','respect']].mean(axis=1)
values_df['high_rating'] = (values_df['overall_score'] >= 4).astype(int)

model8 = sm.ols('overall_score ~ C(group) + C(race) + C(gender) + ses + diversity_gu + diversity_hs', data = values_df).fit()
print(model8.summary())

clean_values_df = values_df.dropna()

X = clean_values_df[['group', 'race', 'gender', 'ses', 'diversity_gu', 'diversity_hs']]
X = pd.get_dummies(X, drop_first=False)

y = clean_values_df['high_rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

logreg = LogisticRegression().fit(X_train, y_train)

print(f"Training set score (accuracy): {logreg.score(X_train, y_train)}")
print(f"Test set score (accuracy): {logreg.score(X_test, y_test)}")

print("Test data predictions")
y_pred_test = logreg.predict(X_test)

confmatrix = confusion_matrix(y_test,y_pred_test)
print(confmatrix)

print(f"accuracy_score: {accuracy_score( y_test, y_pred_test):.2f}")
print(f"precision_score: {precision_score( y_test, y_pred_test):.2f}")
print(f"recall_score {recall_score(y_test,y_pred_test):.2f}")
print(f"f1_score {f1_score(y_test,y_pred_test):.2f}")

print(f"classification_report \n {classification_report(y_test,y_pred_test)}")

#This model is biased toward predicting high ratings
#Why?
#More people likely gave high scores Dataset is imbalanced
#The model achieves moderate accuracy 60% and performs better at identifying individuals with high ratings than low ratings, as reflected by higher recall for the positive class

--------------------------
Random Forest
--------------------------

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf.score(X_test, y_test)

importances = rf.feature_importances_
feature_names = X.columns

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

----------------------------
Feature Importance Chart
----------------------------

feat_imp_df = pd.DataFrame({'Feature': feat_imp.index, 'Importance': feat_imp.values})

norm = plt.Normalize(feat_imp_df['Importance'].min(), feat_imp_df['Importance'].max())

colors = cm.Blues(norm(feat_imp_df['Importance']))

plt.figure(figsize=(8,6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color=colors, edgecolor='black', linewidth=1)
plt.title('Feature Importance', fontsize=14, weight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

#Feature importance results from the machine learning model indicate that socioeconomic status (SES)
#is by far the most influential predictor of overall perception, accounting for over 30% of the model’s importance.
#Diversity perception also plays a role. In contrast, variables such as gender, race, and group membership
#contribute relatively little. This finding aligns with regression results and suggests that perceptions
#of the university’s core values are more strongly associated with socioeconomic background than with
#other demographic characteristics.

---------------------------
More Visualizations
---------------------------

values_df.groupby('ses')[values].mean()

#Higher SES clearly corresponds to higher core value evals, meanwhile lower SES corresponds to lower core value evals.

ses_means = values_df.groupby('ses')[values].mean()

ses_means.plot(marker='o')

plt.title('Core Value Ratings by SES')
plt.xlabel('SES')
plt.ylabel('Average Rating')
plt.legend(title='Values', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

-------------------------
Recommendation
-------------------------

#The university should focus on improving experiences for lower SES groups, particularly in areas where ratings are lowest.
