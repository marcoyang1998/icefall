#!/usr/bin/env perl


# takes as stdin a diagnostics file from icefall by running train.py with --print-diagnostics True.
# Summarizes some information about the eigenvalues of module outputs and parameters.

# this is for analyzing the relative contribution of different modules in conformer layers.

$value_norm = 0;   # stop crash for badly formed input
$one_norm = 1.0e-20;   # stop crash for badly formed input
while (<>) {
  if (m/, size=.+, value.*.+norm=(\S+),/) {
    $value_norm = $1;
  }
  if (m/, size=(\d+), abs.*.+mean=(\S+),/) {
    $size = $1;
    $one_norm = $size * $2;
    if ($one_norm < 1.0e-20) { $one_norm = 1.0e-20; }
  }
  if (m/module=(\S+),.+(dim=.+, size=.+), eigs .+ (\S+) (\S+) (\S+)\], norm=(\S+), mean=(\S+), rms=(\S+)/) {
    $module = $1;
    $dim_and_size = $2;
    $next_next_largest_eig = $3;
    $next_largest_eig = $4;
    $largest_eig = $5;
    $eigs_norm = $6;
    # the eigenvalues printed out are actually the square roots of the eigenvalues of the variance.
    # To get the trace of the variance, we need the rms of the sqrt-eigs (which is the mean of the
    # eigs of the variance), times the number of dimensions i.e. $size.
    $norm_ratio = ($8 * $size) / $one_norm;
    $rms_over_mean = $8 / $7;

    if ($eigs_norm == 0) {
      print("WARN: eigs_norm = 0: $_");
      continue;
    }

    # may not really be next-largest if percentiles were printed, would be 10th-percentile
    # eig.
    if ($eigs_norm - $value_norm > 0) {
      $next_next_largest_ratio = sprintf("%.3f", $next_next_largest_eig / ($eigs_norm - $value_norm));
      $next_largest_ratio = sprintf("%.3f", $next_largest_eig / ($eigs_norm - $value_norm));
    } else {
      #print("eigs-norm=$eigs_norm, value-norm=$value_norm\n");
      $next_next_largest_ratio = 0.0;
      $next_largest_ratio = 0.0;
    }
    $mean_ratio = sprintf("%.3f", $value_norm / $eigs_norm);
    $top_ratio = sprintf("%.3f", $largest_eig / $eigs_norm);
    $rms_over_mean = sprintf("%.3f", $rms_over_mean);
    $norm_ratio = sprintf("%.3f", $norm_ratio);
    print("module=$module, $dim_and_size, norm=$eigs_norm, next-next-largest-ratio=$next_next_largest_ratio, next-largest-ratio=$next_largest_ratio, mean_ratio=$mean_ratio, 2norm/1norm=$norm_ratio, top_ratio=$top_ratio, rms_over_mean=$rms_over_mean\n");
  }
}
