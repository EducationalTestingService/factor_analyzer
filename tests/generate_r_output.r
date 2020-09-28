# install packages, if they're not already installed
packages <- c('argparse', 'psych', 'GPArotation')
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  install.packages(new_packages)
}

library('argparse')
library('psych')

# mapping list for fit methods
mapping1 <- list(minres = 'uls',
                 mle = 'ml')

# mapping list for rotations
mapping2 <- list(geominT = 'geomin_ort',
                 geominQ = 'geomin_obl')

# argument parser
parser <- ArgumentParser(description='Fit some factor models')

parser$add_argument('-n', '--n_factors', type='integer', nargs='+',
                    default=2, help='integer(s) specifying the number of factors')

parser$add_argument('-f', '--fit_methods', type='character', nargs='+',
                    default='minres', help='Fit method(s)')

parser$add_argument('-r', '--rotations', type="character", nargs='+',
                    default='promax', help='Rotaton(s)')

parser$add_argument('-t', '--test_file', type="character", nargs='+',
                    default='test02.csv', help='Test file')



# parse the arguments into a list
args <- parser$parse_args()

# get the input path and directory
path <- args$test_file
dir <- dirname(path)
filename <- basename(path)

# read in the data
df <- read.csv(path)

# loop through all of the conditions
for (n in args$n_factors) {
    for (fm in args$fit_methods) {
        for (rot in args$rotations) {

          # fit the factor model
          res <- fa(cor(df),
                    nfactors = n,
                    n.obs = nrow(df),
                    fm = fm,
                    rotate = rot)

          # get the correct method and rotation names
          fm_name <- mapping1[[fm]]
          fm_name <- if (length(fm_name) == 0) fm else fm_name

          rot_name <- mapping2[[rot]]
          rot_name <- if (length(rot_name) == 0) rot else rot_name

          # write out the loadings
          loadings_file <- paste('loading',
                                 fm_name,
                                 rot_name,
                                 as.character(n),
                                 filename,
                                 sep='_')
          loadings_file <- file.path(dir, loadings_file)
          write.csv(res$loadings, loadings_file)

          # write out the communalities
          communalities_file <- paste('communalities',
                                      fm_name,
                                      rot_name,
                                      as.character(n),
                                      filename,
                                      sep='_')
          communalities_file <- file.path(dir, communalities_file)
          write.csv(res$communalities, communalities_file)
        }
    }
}