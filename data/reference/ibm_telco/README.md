# IBM Telco Customer Churn reference data

Downloaded on 2026-10-26 following the user's instruction to obtain the public dataset while they check organizer data requirements later.

- Publisher/source repository: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d).
- Original file: [Telco-Customer-Churn.csv](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv).
- Local file: Telco-Customer-Churn.csv.
- Size: 970,457 bytes.
- SHA-256: 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91.
- Observed rows: 7,043.
- Observed columns: 21, including customerID and Churn; therefore 19 raw predictor columns after excluding both.
- Churn labels: Yes 1,869; No 5,174.
- Whitespace-only TotalCharges: 11 rows. Preserve as missing after numeric parsing; do not silently discard these customers.
- Repeated customerID groups: zero.

This is a public, fully labelled reference table. It is not verified as the organizer's training file, and no official hidden-test file, sample submission, or portal split has been obtained. Do not describe a locally generated holdout as the organizer's hidden test. Dataset profiling has been performed; no model has been trained.

Keep this source immutable. If official files arrive, retain their separate provenance and use their training membership and submission contract. Do not augment organizer training with public labels for organizer-held-out customers, or claim independent hidden-test performance for rows already used in training. An official dataset change does not silently reset the experiment budget.
