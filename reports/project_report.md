# Retail Marketing Analytics on the Census Income Dataset
### Predicting high earners and building six marketing personas

*Project report prepared April 2026.*
*Data source: 1994-1995 U.S. Current Population Survey, 199,523 individual records with 40 demographic and employment characteristics, representing approximately 347 million people after applying population weights.*

---

## 1. What we set out to do

You asked us to build two things from the same dataset:

1. A prediction model that looks at a person's demographic and work profile and estimates whether they earn above or below $50,000 per year. This lets your marketing team target high-earner offers to the right people instead of sending them to everyone.
2. A segmentation model that sorts the entire population into meaningful groups your marketing team can use as personas, each with its own messaging strategy, product mix, and outreach channel.

We built both models from the same data pipeline, and every number you see in this report reflects the actual U.S. population rather than the raw survey sample. We made this choice because your campaigns will reach real people, and the survey sample alone does not perfectly mirror who those people are. The population weights correct for that gap.

These two models serve different purposes, and that is why we kept them separate rather than combining them into one:

- The prediction model acts like a scorer. It tells you which specific individuals are most likely to be high earners, and it ranks them from most likely to least likely. Think of it as your targeting list.
- The segmentation model acts like a lens. It tells you what broad types of people exist in the population, regardless of their income. Think of it as your persona library for creative direction, product selection, and channel planning.

A single model trying to do both jobs would do neither well. Keeping them separate means the prediction model stays sharp and the personas stay clear.

---

## 2. Understanding the data

**Size and shape.** The dataset contains 199,523 individual records with 40 characteristics each, split roughly evenly between 1994 and 1995 survey years. Only about 6.2% of people in the dataset earn above $50,000. When we apply the population weights, that number moves to 6.4%. Either way, high earners are rare -- roughly 1 in 16 people. This imbalance is important because it shapes how we build and measure the model.

**Why the population weights matter.** Each record in this dataset carries a weight that tells us how many people in the U.S. population that one surveyed person represents. These weights range from 38 to over 18,000, with a middle value around 1,600. Added together, they total about 347 million, which closely matches the U.S. population at that time. Without these weights, every measurement would reflect the survey's sampling design rather than the population your marketing actually reaches. We use them throughout -- during model training and during evaluation -- so every result in this report speaks to the real world.

**"Not in universe" is meaningful, not missing.** Thirteen of the 40 characteristics contain large numbers of records marked "Not in universe." In Census Bureau language, this means the question did not apply to that person. Children, for example, have no occupation code. People who did not move have no migration data. Treating these as blanks and filling them in with guesses would destroy useful information. Someone marked "Not in universe" for occupation is telling us something real about their life situation. We preserved these markings as their own category throughout the analysis.

**Genuinely missing data is limited.** The migration-related columns are about 50% blank because they were only collected from a portion of survey participants. Country of birth is 1-3% blank. Hispanic origin and previous state of residence are less than half a percent blank. Our modeling approach handles these gaps naturally without needing us to fill them in with estimates.

**Duplicate records are real.** About 30,000 records share identical profiles. In a survey this large, two households with the same characteristics are not a data error. They are evidence of how common certain profiles are. Removing them would systematically undercount the most typical types of people.

### What the data tells us before modeling

Figures 1 through 4 below show the most important patterns we found during our initial exploration.

- **Age matters enormously (Figure 1).** The chance of earning above $50,000 starts near zero for people under 18, climbs steadily, peaks around 17% for people aged 45 to 54, and drops back to about 4% after age 65. Age is where most of the predictive power lives.

- **Education is the strongest single predictor (Figure 2).** People with professional degrees (doctors, lawyers, dentists) have above a 50% chance of being high earners. People who did not finish high school sit below 2%. The relationship is steady and consistent: more formal education reliably corresponds to higher income rates.

- **Sex and race show large historical gaps (Figure 3).** Males are roughly four times more likely than females to earn above $50,000 (10.3% versus 2.6%). Asian/Pacific Islander and White respondents both sit above 7%, while Black and Native American respondents fall below 4%. These reflect the income distributions of 1994 and 1995, not current conditions. We discuss the implications of this in Section 7.

- **The population weights follow a wide spread (Figure 4).** Some people in the survey represent 3 to 4 times as many real-world individuals as others. Using these weights during training meaningfully changes which groups the model pays attention to.

![Figure 1 -- Income rate by age group](figures/eda_income_by_age.png)
![Figure 2 -- Income rate by education level](figures/eda_income_by_education.png)
![Figure 3 -- Income rate by sex and race](figures/eda_income_by_sex_race.png)
![Figure 4 -- Distribution of survey weights](figures/eda_weight_distribution.png)

---

## 3. How we prepared the data

Our guiding principle was simple: let the model see the real data. With a dataset this size and a modern modeling approach, heavy-handed data manipulation tends to hurt more than it helps. The gains come from preserving information, not from engineering new features on top.

Here is what we did, step by step:

- **Cleaned up formatting.** Every value in the raw file had a leading space character, which would silently turn "Male" and " Male" into two different categories. We stripped those on load.
- **Handled missing values carefully.** Question marks in the data became genuine blanks. "Not in universe" entries stayed as their own meaningful category. Different kinds of absence got different treatment.
- **Assigned each column its correct role.** Some columns like "detailed industry code" and "detailed occupation code" look like numbers but are actually category labels (government classification codes). Treating them as numbers would force the model to assume that code 200 is somehow "twice as much" as code 100, which is nonsense. We flagged these as categories so the model handles them properly.
- **Let the model work with categories directly.** Our primary model (Histogram Gradient Boosting) can work with category labels natively, so we did not need to convert them into numbers or expand them into columns of ones and zeros. This preserves the original meaning of every category and keeps things simpler.
- **Split the data 80/20 for training and testing.** We set aside 20% of the data as a held-out test set that the model never sees during training, so our performance numbers reflect how the model would work on new, unseen people. We made sure both halves have the same proportion of high earners. We also ensured that rare categories appearing on only one side do not cause problems at prediction time.
- **Built a separate simpler model as a baseline.** For comparison, we also trained a basic logistic regression model. This required converting categories into numeric form, which we did by expanding each category into its own column and grouping rare categories together to keep things manageable.

We did not use any artificial balancing techniques like oversampling or synthetic data generation. Instead, we let the population weights do this work naturally. The survey weights already tell the model which people represent more of the population, which is both a cleaner approach and produces probability estimates you can trust at face value (verified in Section 5).

---

## 4. The prediction model

### 4.1 Which models we evaluated

We tested two approaches side by side:

**A basic linear model (our baseline).** This is a logistic regression, one of the simplest prediction tools available. Think of it as drawing a straight line through the data to separate high earners from everyone else. We used it to answer the question: "How well can a simple approach do?"

**Histogram Gradient Boosting (our production model).** This is a more sophisticated approach that builds hundreds of small decision trees, each one learning from the mistakes of the previous ones. It handles mixed data types, missing values, and complex patterns naturally. We chose this specific implementation (from the scikit-learn library) over similar tools like LightGBM and XGBoost because all three perform equivalently on data this size, but this one requires no extra software installation. It works right out of the box on any machine, which makes the deliverable simpler and more reliable.

### 4.2 How we tuned the model

The model has several settings that affect how it learns. Rather than picking these by hand, we used an automated search tool called Optuna that intelligently explores different combinations. We ran 25 different configurations, testing each one on three separate folds of the training data, and optimized for the metric that matters most to your use case: how well the model identifies the top of the ranking (technically called precision-recall area under the curve, or PR AUC). At a 6% base rate, this metric is far more relevant than the commonly used ROC AUC because it focuses on the head of the list -- exactly where marketing dollars go.

The six settings we searched over and their ranges:

| Setting              | Range tested           |
|----------------------|------------------------|
| Learning rate        | 0.02 to 0.15           |
| Number of iterations | 200 to 900             |
| Tree complexity      | 16 to 96 leaf nodes    |
| Minimum group size   | 10 to 120 samples      |
| Regularization       | 0.001 to 5.0           |
| Feature sampling     | 60% to 100%            |

The best configuration achieved a PR AUC of 0.6669 in cross-validation. The improvement over our hand-picked starting point (0.6596) was modest. This tells us something valuable: this modeling family has nearly extracted all the predictive signal available in this data. There is limited headroom left, which is useful to know for planning purposes.

### 4.3 Setting the decision cutoff

A prediction model outputs a probability score between 0 and 1. To turn that into an actionable yes-or-no decision, we need a cutoff: everyone above this score gets the "high earner" treatment, everyone below does not. We selected this cutoff by finding the value that produces the best balance of correctly identifying real high earners without flagging too many non-high-earners. Critically, we did this using cross-validation predictions from the training data, not from the test set. This prevents the subtle but important error of using test data to make model decisions, which would make our reported results artificially optimistic.

The selected cutoff is 0.27. We also report results at two other useful operating points: the top 5% of the ranked list (for high-value offers) and the top 10% (for broader mailer campaigns).

### 4.4 Results

**Performance on the held-out test set (39,905 people, representing about 69 million Americans):**

| Metric                                   | Basic Model | Production Model |
|------------------------------------------|------------:|-----------------:|
| Ranking quality (ROC AUC)                |      0.9461 |       0.9542     |
| Precision-recall quality (PR AUC)        |      0.6264 |       0.6931     |
| Probability accuracy (Brier score)       |      0.0362 |       0.0329     |
| Balanced accuracy at cutoff (F1 at 0.27) |      0.5915 |       0.6225     |
| Hit rate in top 5% of list               |      0.6514 |       0.7105     |
| Hit rate in top 10% of list              |      0.4711 |       0.4837     |
| Lift vs random in top 10%               |       7.30x |        7.49x     |

**What this means for your campaigns.** If your team takes the top 10% of people ranked by this model and sends them a high-income offer, 48% of them will genuinely be high earners. Compare that to a random mailing, where only 6.4% would be high earners. That is a 7.5 times improvement. If you narrow the list to the top 5%, the hit rate rises to 71%.

We verified these results are stable by running five-fold cross-validation. The numbers vary by less than 0.004 from fold to fold, confirming the model is not memorizing the training data -- it genuinely generalizes.

![Figure 5 -- How well the model ranks people (ROC curves)](figures/clf_roc.png)
![Figure 6 -- Precision vs recall tradeoff](figures/clf_pr.png)

**Can you trust the probability scores? (Figure 7)** Yes. The calibration chart shows that when the model says someone has a 30% chance of being a high earner, roughly 30% of those people actually are. This alignment holds across the full range of scores, with the largest deviation under one percentage point. This is unusually good calibration, and it means your team can use the raw probability scores directly in business calculations -- for example, multiplying the score by the expected profit per conversion to decide who is worth contacting. No extra adjustment step is needed.

![Figure 7 -- Calibration: predicted vs actual probabilities](figures/clf_calibration.png)

**Where the model gets it right and wrong (Figure 8).** At our chosen cutoff of 0.27, the model correctly identifies about 66% of all high earners while wrongly flagging only 3% of non-high-earners. The 34% of high earners it misses are almost all borderline cases whose scores fall between 0.15 and 0.27. Lowering the cutoff to catch them would cost two to three times as much in false positives.

![Figure 8 -- Confusion matrix showing correct and incorrect predictions](figures/clf_confusion.png)

**Which characteristics drive the predictions (Figure 9).** We measured each feature's contribution by randomly scrambling its values and observing how much the model's performance drops. Age dominates, consistent with the strong age pattern we saw in the data exploration. Weeks worked per year, education level, capital gains, tax filer status, sex, and detailed occupation follow. The bottom 15 features -- including 5 of the 7 migration-related columns -- contribute almost nothing. The model could be simplified to roughly 25 features with no measurable loss in performance.

![Figure 9 -- Feature importance: which characteristics matter most](figures/clf_feature_importance.png)

### 4.5 Performance across demographic groups

We checked whether the model works equally well across different demographic groups:

| Group  | Subgroup | High-earner rate | Hit rate | Catch rate | Ranking quality |
|:-------|:---------|-----------------:|---------:|-----------:|----------------:|
| Sex    | Male     |            10.5% |    0.615 |      0.720 |           0.948 |
| Sex    | Female   |             2.6% |    0.473 |      0.412 |           0.940 |
| Race   | White    |             7.1% |    0.597 |      0.669 |           0.953 |
| Race   | Asian/PI |             8.8% |    0.653 |      0.662 |           0.955 |
| Race   | Black    |             2.4% |    0.464 |      0.426 |           0.950 |

The ranking quality is stable across all groups (ranging from 0.91 to 0.97), meaning the model ranks people within each group equally well. Where we see differences is in hit rate and catch rate, and these differences are driven by the underlying income rates in each group, not by the model performing worse for certain groups. When a group has a lower overall income rate, a single cutoff naturally catches fewer of its high earners. This is an inherent property of using one cutoff across groups with different income distributions, and we discuss it further in Section 7.

---

## 5. The segmentation model

### 5.1 Why we chose K-Prototypes

Your data contains a mix of number-based characteristics (like age and wage) and category-based characteristics (like occupation and marital status). Most grouping algorithms are designed for one type or the other, not both. K-Prototypes handles both naturally: it measures numerical similarity using standard distance and categorical similarity by checking whether categories match. This preserves the meaning of each data type.

We considered and rejected two alternatives:

- **Converting categories to numbers and using standard grouping (K-Means).** This would treat "Divorced" and "Widowed" as equally different from "Never married," losing the real-world relationships between categories. It also produces groups that are hard to describe in plain language.
- **A probability-based approach on compressed data (Gaussian Mixture Models).** This rotates the data into abstract mathematical dimensions that lose their real-world meaning. Useful in some contexts, but your marketing team needs groups described in terms they can act on.

One limitation: K-Prototypes does not support population weights during the grouping process. We handled this by grouping the raw data (which gives stable, well-defined groups) and then describing each group using the population weights. This means the population sizes, income rates, and comparisons you see below correctly reflect the U.S. population.

### 5.2 Which characteristics we used

We deliberately used a smaller set of characteristics for segmentation than for prediction, because the goals are different. The prediction model should use every scrap of information that improves accuracy. The segmentation model should use only characteristics your marketing team can understand, act on, and recognize on a persona card.

We kept: age, education, marital status, occupation, industry, household composition, employment status, hourly wage, capital gains and losses, dividend income, and tax filing status. We dropped migration codes, detailed industry and occupation recodes, and other technical survey fields that add noise without adding marketing clarity.

### 5.3 How we chose six groups

We tested group counts from 3 to 8 on a 20,000-person sample and charted the results (Figure 10). A purely mathematical approach would suggest 4 groups, which cleanly separates children, retirees, and two working-age clusters. We chose 6 groups for a business reason: at 4 groups, two tiny but extremely valuable segments (the self-employed professional elite and the affluent retirees) get absorbed into the broad working-professional cluster and lose their distinctive identity. These two micro-segments represent less than half a percent of the population but account for a disproportionate share of any premium product's total addressable value. Losing them in a larger group would be mathematically tidy but commercially costly.

![Figure 10 -- Cost curve for different numbers of groups](figures/seg_k_sweep.png)

### 5.4 The six personas

Below is a profile of each segment, ordered from highest income rate to lowest. "Lift" means how many times more likely someone in that segment is to be a high earner compared to a randomly chosen adult.

---

**Segment C0 -- "Self-Employed Professional Elite"** *(0.2% of the population; 88% are high earners; 13.8x lift)*

These are highly educated professionals who run their own practices -- doctors, lawyers, dentists, and veterinarians are 27 times overrepresented here compared to the general population. Their investment income hits the survey's maximum recordable value of $99,999. Average age is 46. They work full-year schedules.

**How to reach them.** Luxury offers, premium professional services, and high-ticket business purchases. These are price-insensitive decision-makers. Reach them through direct mail and professional network advertising, not mass media.

---

**Segment C2 -- "Affluent Retired Investors"** *(0.2% of the population; 71% are high earners; 11.0x lift)*

Older individuals (average age 59) with substantial investment income -- average dividend income of $37,000 and very high capital gains. A tiny group by headcount but disproportionately wealthy. Most are married.

**How to reach them.** Wealth preservation products, premium travel, high-end healthcare, estate planning services, and luxury goods positioned around legacy rather than aspiration. Use trusted direct mail and long-form content.

---

**Segment C1 -- "Working Professionals on the Rise"** *(2.0% of the population; 30% are high earners; 4.8x lift)*

Prime working-age professionals with long hours and active investment portfolios (their capital losses suggest they are actively managing investments). Professional degrees are 4 times overrepresented. Think of them as the younger, larger sibling of Segment C0 -- still building their careers rather than running established practices.

**How to reach them.** Mid-to-premium products, career development services, financial planning, and high-end consumer electronics. Use targeted digital advertising and professional publications.

---

**Segment C3 -- "Working and Middle America"** *(46.7% of the population; 11% are high earners; 1.7x lift)*

This is nearly half the population. Prime working age (average 38), full-year employment, wage earners. Education spans high school through some college. This is your volume segment -- the audience that most mass marketing is designed for.

**How to reach them.** Mass-market value propositions, financing options, and loyalty programs. The 11% high-earner rate means there is meaningful variety within this group. This is exactly the segment where the prediction model earns its keep: it can subdivide these 46.7% of people into likely high earners and likely non-high-earners for differentiated treatment.

---

**Segment C5 -- "Retired, Out of the Workforce"** *(18.2% of the population; 1.6% are high earners)*

Average age 64, almost zero weeks worked. Widowed individuals are 4.4 times overrepresented. Very little wage income, modest investment income.

**How to reach them.** Healthcare, insurance, housing, groceries, and travel for those who can afford it. Generally price-sensitive and loyal to established brands. Reach them through television, mail, and increasingly through digital channels. Not a premium target, but a large and stable audience.

---

**Segment C4 -- "Children Under 18"** *(32.8% of the population; near-zero high earners)*

About 69% of this group has "Children" as their education level, and 72% are classified as "Child under 18, never married." They are not directly marketable, but they are critical for household-level decisions. Families with children are among the top three spending segments in retail.

**How to reach them.** Reach them through their parents (identifiable in Segment C3 by the number of children in the household). Focus on kid-adjacent products, back-to-school campaigns, and family entertainment offers.

---

![Figure 11 -- High-earner rate by segment](figures/seg_income_by_cluster.png)

### 5.5 Using both models together

The two models work best in combination:

1. **Start with the segments.** Assign each customer to one of the six personas. This tells you what kind of messaging to use, which products to feature, and which channel to reach them through.
2. **Then apply the prediction scores.** Within each segment, rank customers by their likelihood of being high earners. This tells you how much to spend on each person and whether they get the premium treatment or the standard one.

Here is a concrete example -- say you are launching a premium credit card:

- Segments C0 and C2 get a white-glove direct mail invitation with no application friction. The prediction model is almost unnecessary here because both groups are majority high earners.
- Segment C1 gets the premium offer, but only the top 20% by prediction score. That is roughly 1% of the total population, with close to a 70% acceptance-eligibility rate.
- Segment C3 members with a prediction score above 0.6 get a mid-tier offer. This is where the bulk of your directed spend should go.
- Segments C5 and C4 get offers appropriate to their life stage, or no offer at all. A premium card pitch would be wasteful here.

This two-model approach is more valuable than either model alone. The segmentation tells you what to say. The prediction model tells you who to say it to.

---

## 6. Notable findings

- **Investment income is capped in the original data.** Capital gains max out at $99,999 in this dataset -- that is a recording limit, not a real ceiling. Segment C0 shows mean capital gains of exactly $99,999 because many of its members hit this cap. Any future analysis using raw capital gains values should treat $99,999 as "at least this much" rather than as an exact figure. Our model handles this naturally because it groups values into ranges rather than using exact amounts.

- **The year of the survey does not matter.** When we tested whether knowing the survey year (1994 vs 1995) helps the model, it contributed nothing. Scrambling this feature had zero effect on performance. This is reassuring: it means the patterns are stable across both years and the model is not relying on year-specific quirks.

- **Migration data adds almost nothing.** Five of the seven migration-related columns rank at the very bottom of our feature importance list. About half the records have no migration data at all (because it was only collected from a subset), and even where data exists, the signal is weak.

- **Three characteristics carry most of the weight.** Sex, education, and age together account for roughly half of the model's total predictive power. A stripped-down model using only five features (age, weeks worked, education, capital gains, and sex) would likely retain most of the headline performance. This is worth knowing if a future application requires a simpler model.

- **The highest-value segment disappears at four groups.** When the math suggests four groups, it merges the self-employed professional elite (Segment C0) into the broad working class. Mathematically reasonable. Commercially, it would mean losing sight of a group where 88% are high earners. This is why human judgment on the number of groups is essential.

---

## 7. Fairness and responsible use

This dataset records race, sex, and national origin, and the model reproduces the income patterns that existed in the 1994-1995 U.S. labor market. Here is what that means in practice:

- The model flags males as likely high earners about five times more often than females (12.3% versus 2.3%). Both rates closely match the actual historical rates (10.5% and 2.6%), so the model is not creating bias. But it is not correcting historical bias either -- it mirrors the world as it was in 1994.
- The model catches fewer high earners in groups that have lower income rates overall (women, Black respondents, American Indian respondents). This is a mathematical consequence of applying one cutoff across groups with different starting rates, not a flaw in how the model ranks people within each group.
- The ranking quality is consistent across all groups (0.91 to 0.97), meaning the model is equally good at telling apart high earners from non-high-earners within each group. The differences show up only in the absolute cutoff, not in the ranking itself.

**What we recommend:**

1. Do not use this model for any decision affecting credit, insurance pricing, employment, housing, or eligibility. The data is 30 years old, the outcome itself is a legally sensitive attribute in many contexts, and the model directly uses race, sex, and national origin as inputs.
2. For discretionary marketing use (catalog mailers, promotional offers, product recommendations), the model is defensible. Even so, consider removing sex, race, and Hispanic origin from the inputs. This would reduce precision-recall performance by about 2-3% but produce a model that does not directly use protected attributes. Monitor opt-out rates and response rates by demographic group for any campaign using this scoring.
3. Apply the same scrutiny to the segmentation model. Segments C0 through C5 have very different demographic compositions. For example, the retired segment skews female due to mortality differences between sexes. A persona profile is not inherently neutral.

---

## 8. Recommendations and next steps

**What you can do now:**

1. Deploy the prediction model as a scoring service. The saved model file accepts the 40 data columns and returns a calibrated probability score. No preprocessing step is needed on your end.
2. Implement the segmentation as a lookup table based on the 12 segmentation characteristics. The saved segmentation model can also score new customers directly, but a lookup table is faster to integrate with most CRM systems.
3. Use the two models together as described in Section 5.5: segment first, then score within each segment.
4. Treat the default cutoff of 0.27 as a starting point. The right cutoff depends on your campaign economics. If a wasted contact costs $2 and a successful conversion returns $40, the optimal cutoff drops below 0.27.

**What we would tackle next:**

- **Update the data.** This 1994-1995 survey is an excellent foundation for methodology, but any production deployment needs current data. Education levels, household structures, workforce participation, and wage distributions have all shifted meaningfully in 30 years.
- **Remove sensitive attributes** from the model inputs if deploying in production, replacing race, sex, and Hispanic origin with alternatives that do not directly encode protected characteristics.
- **Build a cost-based evaluation.** Our current cutoff maximizes a statistical measure. Your real-world cutoff should reflect campaign economics -- what a false positive costs you versus what a true positive earns you.
- **Explore alternative segmentation methods** that can discover small, dense clusters that our current approach might merge into larger groups. This could reveal additional high-value niches.
- **Break up the largest segment.** With nearly 47% of the population in Segment C3, your marketing team will eventually need finer distinctions within that group.

---

## 9. Questions for your team

1. **What does a wasted contact cost versus a successful conversion?** This directly determines the right prediction cutoff for each campaign.
2. **Will this model inform eligibility decisions, or purely discretionary offers?** The answer changes the fairness requirements by an order of magnitude.
3. **Should the segmentation be refreshed periodically or treated as a fixed set of personas?** If the former, we should build an automated re-segmentation pipeline. If the latter, the current output is final.
4. **Does your marketing team have existing persona definitions?** Mapping our six segments to your internal names would make this output immediately usable in your existing workflows.

---

## References

Full reference list available in reports/references.md. Key sources consulted: Chen and Guestrin (2016) on gradient boosting methodology; Huang (1998) on the K-Prototypes clustering algorithm; Niculescu-Mizil and Caruana (2005) on probability calibration for classifiers; scikit-learn and Optuna library documentation; U.S. Census Bureau Current Population Survey technical documentation for survey weights and data recording rules; KDD Cup 1999 Census-Income dataset description as the original source of this data extract.
