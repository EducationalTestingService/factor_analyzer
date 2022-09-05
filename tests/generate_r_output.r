# install packages, if they're not already installed
packages <- c('argparse', 'psych', 'GPArotation')
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  install.packages(new_packages,
                   repos = "http://cran.us.r-project.org")
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

parser$add_argument('-n', '--n_factors', type='integer',
                    default=2, help='integer(s) specifying the number of factors')

parser$add_argument('-f', '--fit_methods', type='character',
                    default='minres', help='Fit method(s)')

parser$add_argument('-r', '--rotations', type="character",
                    default='promax', help='Rotaton(s)')

parser$add_argument('-t', '--test_file', type="character",
                    default='test02.csv', help='Test file')

parser$add_argument('-o', '--output_dir', type="character",
                    default=NULL, help='Output directory')


# parse the arguments into a list
args <- parser$parse_args()

# get the input path and directory
path <- args$test_file
if (is.null(args$output_dir)) {
  dir <- dirname(path)
} else {
  dir <- args$output_dir
  dir.create(dir, showWarnings = FALSE)
}

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

          # get outputs
          loadings <- res$loadings;
          values <- res$values;
          evalues <- res$e.values;
          uniquenesses <- res$uniquenesses;
          communalities <- res$communalities;

          info <- list('loading' = loadings,
                       'value' = values,
                       'evalues' = evalues,
                       'uniquenesses' = uniquenesses,
                       'communalities' = communalities)
          for (name in names(info)) {
            df_temp <- info[[name]]
            out <- paste(name,
                         fm_name,
                         rot_name,
                         toString(n),
                         filename,
                         sep='_')
            out_file <- file.path(dir, out)
            write.csv(df_temp, out_file)
          }
        }
    }
}
