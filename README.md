# project steps

## preprocessing

### removing cloums with no need

1. rowid
2. kepid
3. kepoi_name
4. kepler_name
5. koi_pdisposition
6. koi_score
7. koi_teq_err1
8. koi_teq_err2

### moving the problem form 3 way classification to only binary classification 

- Limiting the values of the target column: Only the values candidate and confirmed
  were left. Given the presence of false-positive data within the column, it was deemed
  essential to focus solely on instances classified as either candidate or confirmed
  exoplanets. Accordingly, rows containing false-positive values were removed from
  the dataset to ensure the accuracy and integrity of the assessment  

### imputation of koi_tce_delivname

### stander scaling

