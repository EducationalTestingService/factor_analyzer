"""
Factor analysis command line script.

:author: Jeremy Biggs (jbiggs@ets.org)

:date: 12/13/2017
:organization: ETS
"""

import os
import argparse
import logging
import pandas as pd

from factor_analyzer.factor_analyzer import FactorAnalyzer


def main():
    """ Run the script.
    """

    # set up an argument parser
    parser = argparse.ArgumentParser(prog='factor_analyzer.py')
    parser.add_argument(dest='feature_file',
                        help="Input file containing the pre-processed features "
                             "for the training data")
    parser.add_argument(dest='output_dir', help="Output directory to save "
                                                "the output files", )
    parser.add_argument('-f', '--factors', dest="num_factors", type=int,
                        default=3, help="Number of factors to use (Default 3)",
                        required=False)

    parser.add_argument('-r', '--rotation', dest="rotation", type=str,
                        default='none', help="The rotation to perform (Default 'none')",
                        required=False)

    parser.add_argument('-m', '--method', dest="method", type=str,
                        default='minres', help="The method to use (Default 'minres')",
                        required=False)

    # parse given command line arguments
    args = parser.parse_args()

    method = args.method
    factors = args.num_factors
    rotation = None if args.rotation == 'none' else args.rotation

    file_path = args.feature_file

    if not file_path.lower().endswith('.csv'):
        raise ValueError('The feature file must be in CSV format.')

    data = pd.read_csv(file_path)

    # get the logger
    logger = logging.getLogger(__name__)
    logging.setLevel(logging.INFO)

    # log some useful messages so that the user knows
    logger.info("Starting exploratory factor analysis on: {}.".format(file_path))

    # run the analysis
    analyzer = FactorAnalyzer()
    analyzer.analyze(data, factors, rotation, method)

    # create paths to loadings loadings, eigenvalues, communalities, variance
    path_loadings = os.path.join(args.output_dir, 'loadings.csv')
    path_eigen = os.path.join(args.output_dir, 'eigenvalues.csv')
    path_communalities = os.path.join(args.output_dir, 'communalities.csv')
    path_variance = os.path.join(args.output_dir, 'variance.csv')

    # retrieve loadings, eigenvalues, communalities, variance
    loadings = analyzer.loadings
    eigen, _ = analyzer.get_eigenvalues()
    communalities = analyzer.get_communalities()
    variance = analyzer.get_factor_variance()

    # save the files
    logger.info("Saving files...")
    loadings.to_csv(path_loadings)
    eigen.to_csv(path_eigen)
    communalities.to_csv(path_communalities)
    variance.to_csv(path_variance)


if __name__ == '__main__':

    main()
