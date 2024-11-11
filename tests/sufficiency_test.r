data = read.csv("tests/data/test01.csv")

factor_range = 1:15
statistic = rep(0, length(factor_range))
dof = rep(0, length(factor_range))
pvalue = rep(0, length(factor_range))
for (i in 1:length(factor_range)) {
  res = factanal(data, factors=factor_range[i], rotation="none")
  statistic[i] = res$STATISTIC
  dof[i] = res$dof
  pvalue[i] = res$PVAL
}

f = data.frame(n_factors=factor_range, statistic=statistic, df=dof, pvalue=pvalue)
write.csv(f, "tests/expected/test01/sufficiency_ml_none_15_test01.csv", row.names=FALSE, quote=FALSE)