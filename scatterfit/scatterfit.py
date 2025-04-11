import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from patsy import dmatrices
from mpl_toolkits.axes_grid1 import make_axes_locatable


class ScatterFit:

    # Define default color scheme
    DEFAULT_COLOR_CYCLE = [
        (0.192, 0.443, 0.651),  # Blue-ish
        (0.824, 0.0, 0.0),  # Red-ish
        (0.059, 0.537, 0.004),  # Green-ish
        (1.0, 0.498, 0.055),  # Orange-ish
        (0.663, 0.227, 0.894),  # Purple-ish
        (0.161, 0.851, 0.906),  # Cyan-ish
        (0.98, 0.933, 0.086),  # Yellow-ish
        (0.871, 0.451, 0.196)  # Brown-ish
    ]
    DEFAULT_SCATTER_COLOR = DEFAULT_COLOR_CYCLE[0]
    DEFAULT_FIT_COLOR = DEFAULT_COLOR_CYCLE[1]
    DEFAULT_GRID_COLOR = "#D3D3D3"
    DEFAULT_TEXT_COLOR = "#555555"
    DEFAULT_EDGE_COLOR = "#555555"

    def __init__(self, data, y_var, x_var, text_scale=1.0, marker_size=None, **kwargs):
        """
        Initialize the ScatterFit object with data and variables.

        Args:
            data (pd.DataFrame): DataFrame containing the data
            y_var (str): Name of the dependent variable column
            x_var (str): Name of the independent variable column
            **kwargs: Optional parameters for customizing the plot
        """
        self.data = data.copy()
        self.y_var = y_var
        self.x_var = x_var
        self.text_scale = text_scale
        self.marker_size = marker_size

        # Process all optional parameters with defaults
        self.options = self._process_options(kwargs)

        # Initialize placeholder for the axes
        self.ax = None
        self.kde_ax = None

        # Prepare data and variables
        self._prepare_data()

    def _process_options(self, kwargs):
        options = {
            # Binning
            "binned": kwargs.get("binned", False),
            "n_quantiles": kwargs.get("n_quantiles", 30),
            'uni_bins': kwargs.get('uni_bins', None),
            'discrete': kwargs.get('discrete', False),
            'bin_var': kwargs.get('bin_var', None),

            # Fit line
            'fit': kwargs.get('fit', 'linear'),
            'ci': kwargs.get('ci', False),
            'level': kwargs.get('level', 95.0),
            'bw_frac': kwargs.get('bw_frac', 0.666),
            'fitmodel': kwargs.get('fitmodel', None),

            # Grouping
            'by_var': kwargs.get('by_var', None),
            'bymethod': kwargs.get('bymethod', 'stratify'),

            # Control variables
            'controls': [kwargs.get('controls')] if isinstance(kwargs.get('controls'), str) else kwargs.get('controls',
                                                                                                            None),
            'fcontrols': [kwargs.get('fcontrols')] if isinstance(kwargs.get('fcontrols'), str) else kwargs.get(
                'fcontrols', None),

            # Regression parameters
            'regparameters': kwargs.get('regparameters', None),
            'parpos': kwargs.get('parpos', None),
            'parsize': kwargs.get('parsize', None),

            # Appearance
            'mweighted': kwargs.get('mweighted', False),
            'mlabel': kwargs.get('mlabel', None),
            'jitter': kwargs.get('jitter', None),
            'standardize': kwargs.get('standardize', False),
            'colorscheme': kwargs.get('colorscheme', None),
            'xdistribution': kwargs.get('xdistribution', None),
            'xdistrbw': kwargs.get('xdistrbw', None),

            # Axis titles
            'xtitle': kwargs.get('xtitle', None),
            'ytitle': kwargs.get('ytitle', None),

            # Technical
            'weight_var': kwargs.get('weight_var', None),
            'ax': kwargs.get('ax', None)
        }

        # Validate and adjust options
        if options['fit'] not in ['linear', 'quadratic', 'cubic', 'lpoly', 'lowess', 'none']:
            print(f"Warning: Unknown fit type '{options['fit']}'. Using 'linear'.")
            options['fit'] = 'linear'

        if options['bymethod'] not in ['stratify', 'interact']:
            print(f"Warning: Unknown bymethod '{options['bymethod']}'. Using 'stratify'.")
            options['bymethod'] = 'stratify'

        return options

    def _prepare_data(self):
        """Prepare the data for plotting, including handling binary DV and residualization."""
        # Collect required columns
        essential_cols = [self.y_var, self.x_var]
        if self.options['by_var'] and self.options['by_var'] in self.data:
            essential_cols.append(self.options['by_var'])
        if self.options['bin_var'] and self.options['bin_var'] in self.data:
            essential_cols.append(self.options['bin_var'])
        if self.options['mlabel'] and self.options['mlabel'] in self.data:
            essential_cols.append(self.options['mlabel'])
        if self.options['controls']:
            essential_cols.extend([c for c in self.options['controls'] if c in self.data])
        if self.options['fcontrols']:
            essential_cols.extend([f for f in self.options['fcontrols'] if f in self.data])
        if self.options['weight_var'] and self.options['weight_var'] in self.data:
            essential_cols.append(self.options['weight_var'])

        # Use unique column names and check existence
        essential_cols = list(set(c for c in essential_cols if c in self.data))
        self.df = self.data[essential_cols].dropna(subset=[self.y_var, self.x_var]).copy()

        if self.df.empty:
            print("Warning: No valid data points remaining after dropping NAs in y_var/x_var.")
            return

        # Handle binary dependent variable detection
        self.binary_dv = self._detect_binary_dv()

        # Standardize variables
        self.y_plot_var, self.x_plot_var = self._standardize_if_needed()

        # Residualize if control variables are specified
        self._residualize_if_needed()

        # Binning
        self._setup_binning()

    def _detect_binary_dv(self):
        """Detect if the dependent variable is binary and convert if necessary."""
        y_unique_non_na = self.df[self.y_var].dropna().unique()
        if len(y_unique_non_na) == 2:
            # Check if y_var is categorical or object type and convert to numeric
            if self.df[self.y_var].dtype.name in ['category', 'object']:
                print(f"Info: Converting {self.y_var} from {self.df[self.y_var].dtype} to numeric.")
                # First convert to string to handle category type
                y_values = self.df[self.y_var].astype(str)
                # Then map to 0/1 values
                y_min, y_max = sorted(y_values.unique())
                self.df[self.y_var] = y_values.map({y_min: 0, y_max: 1}).astype(float)
            # If y_var is already numeric but not 0/1
            elif not np.all(np.isin(y_unique_non_na, [0, 1])):
                y_min, y_max = y_unique_non_na.min(), y_unique_non_na.max()
                print(f"Info: Binary variable {self.y_var} detected. Mapping {y_min}->0, {y_max}->1.")
                self.df[self.y_var] = self.df[self.y_var].map({y_min: 0, y_max: 1}).astype(float)

            # Adjust / warn about options in binary case
            if not self.options['fitmodel']:
                self.options['fitmodel'] = 'logit'
            if not self.options['binned']:
                print("Warning: Using non-binned scatter for binary DV is usually not informative.")

            return True

        return False

    def _standardize_if_needed(self):
        """Standardize variables if requested."""
        y_plot_var, x_plot_var = self.y_var, self.x_var  # Start with original names

        if self.options['standardize']:
            weight_var = self.options['weight_var']

            for var in [self.y_var, self.x_var]:
                weights = self.df.loc[self.df[var].notna(), weight_var] if weight_var else None
                mean = np.average(self.df[var].dropna(), weights=weights)
                std = np.sqrt(np.average((self.df[var].dropna() - mean) ** 2, weights=weights))

                if std > 1e-9:  # Avoid division by zero
                    self.df[f"{var}_std"] = (self.df[var] - mean) / std
                else:
                    self.df[f"{var}_std"] = 0

            y_plot_var = f"{self.y_var}_std"
            x_plot_var = f"{self.x_var}_std"

        return y_plot_var, x_plot_var

    def _residualize_if_needed(self):
        """Residualize variables if control variables are specified."""
        if not (self.options['controls'] or self.options['fcontrols']):
            return

        # For non-binned data, use standard residualization
        if not self.options['binned']:
            self.df = self._residualize(
                self.df,
                self.y_var,
                self.x_var,
                self.options['controls'],
                self.options['fcontrols'],
                self.options['weight_var']
            )

            self.y_plot_var = f"{self.y_var}_resid"
            self.x_plot_var = f"{self.x_var}_resid"

            # Ensure residualized vars exist even if residualization failed
            if self.y_plot_var not in self.df.columns:
                self.df[self.y_plot_var] = self.df[self.y_var]
            if self.x_plot_var not in self.df.columns:
                self.df[self.x_plot_var] = self.df[self.x_var]
        else:
            # For binned data with controls, implement Cattaneo et al. (2023) approach
            # First make sure bins are created
            self._setup_binning()

            if self.bin_col is None:
                print("Warning: Could not create bins for covariate adjustment. Using standard approach.")
                # Fall back to standard residualization
                self.df = self._residualize(
                    self.df,
                    self.y_var,
                    self.x_var,
                    self.options['controls'],
                    self.options['fcontrols'],
                    self.options['weight_var']
                )
                self.y_plot_var = f"{self.y_var}_resid"
                self.x_plot_var = f"{self.x_var}_resid"
                return

            # Create adjusted variables for y
            self.y_plot_var = f"{self.y_var}_adj"
            # x variable remains the same for binned data
            self.x_plot_var = self.x_var
            self.df[self.y_plot_var] = np.nan

            # Get by groups
            by_var = self.options['by_var']
            by_groups = [None]
            if by_var and by_var in self.df:
                by_groups = sorted(self.df[by_var].dropna().unique())

            # Process each group separately
            for group_val in by_groups:
                # Set up group filter
                if by_var and group_val is not None:
                    group_mask = self.df[by_var] == group_val
                else:
                    group_mask = pd.Series(True, index=self.df.index)

                # Skip if no data in this group
                if not any(group_mask):
                    continue

                # Set up weights
                weight_var = self.options['weight_var']
                weights = self.df.loc[group_mask, weight_var] if weight_var and weight_var in self.df else None

                # Build formula for regression with bin indicators and controls
                formula_parts = []

                # Add bin indicators (categorical)
                formula_parts.append(f"C({self.bin_col})")

                # Add controls
                if self.options['controls']:
                    for ctrl in self.options['controls']:
                        if ctrl in self.df.columns:
                            formula_parts.append(ctrl)

                # Add factor controls
                if self.options['fcontrols']:
                    for fctrl in self.options['fcontrols']:
                        if fctrl in self.df.columns:
                            formula_parts.append(f"C({fctrl})")

                # Build the formula
                formula = f"{self.y_var} ~ " + " + ".join(formula_parts)

                try:
                    # Step 1: Fit joint regression model of y on bins and controls
                    # Per Cattaneo et al. (2023), equation (2.3)
                    if weights is not None:
                        model = smf.wls(formula, data=self.df.loc[group_mask], weights=weights, missing='drop').fit()
                    else:
                        model = smf.ols(formula, data=self.df.loc[group_mask], missing='drop').fit()

                    # Step 2: Extract bin coefficients (β̂)
                    bin_effects = pd.Series(0.0, index=self.df.loc[group_mask].index)

                    # Add intercept if present
                    if 'Intercept' in model.params:
                        bin_effects += model.params['Intercept']

                    # Add bin-specific effects
                    for param_name, value in model.params.items():
                        if param_name.startswith(f"C({self.bin_col})"):
                            # Extract the bin number
                            try:
                                bin_num = param_name.split('[T.')[-1].replace(']', '')
                                # Apply the coefficient to matching observations
                                bin_effects.loc[self.df.loc[group_mask, self.bin_col].astype(str) == bin_num] += value
                            except (IndexError, ValueError) as e:
                                print(f"Warning: Could not extract bin number from {param_name}: {e}")

                    # Step 3: Calculate mean of covariates (w̄)
                    cov_means = {}

                    # For continuous covariates, simple mean
                    if self.options['controls']:
                        for ctrl in self.options['controls']:
                            if ctrl in model.params:
                                if weights is not None:
                                    valid_mask = ~self.df.loc[group_mask, ctrl].isna()
                                    if valid_mask.any():
                                        cov_means[ctrl] = np.average(
                                            self.df.loc[group_mask, ctrl][valid_mask],
                                            weights=weights[valid_mask]
                                        )
                                    else:
                                        cov_means[ctrl] = 0
                                else:
                                    cov_means[ctrl] = self.df.loc[group_mask, ctrl].mean()

                    # For categorical covariates, proportion in each category
                    if self.options['fcontrols']:
                        for fctrl in self.options['fcontrols']:
                            fctrl_params = [p for p in model.params.index if p.startswith(f"C({fctrl})")]
                            for param_name in fctrl_params:
                                # Extract the category
                                try:
                                    category = param_name.split('[T.')[-1].replace(']', '')
                                    # Calculate proportion in this category
                                    if weights is not None:
                                        valid_mask = self.df.loc[group_mask, fctrl].astype(str) == category
                                        if valid_mask.any():
                                            cov_means[param_name] = np.average(
                                                valid_mask,
                                                weights=weights[valid_mask.index]
                                            )
                                        else:
                                            cov_means[param_name] = 0
                                    else:
                                        cov_means[param_name] = (
                                                    self.df.loc[group_mask, fctrl].astype(str) == category).mean()
                                except (IndexError, ValueError) as e:
                                    print(f"Warning: Could not extract category from {param_name}: {e}")

                    # Step 4: Calculate mean covariate effect (w̄'γ̂)
                    mean_cov_effect = 0
                    for param_name, value in model.params.items():
                        if param_name in cov_means:
                            mean_cov_effect += value * cov_means[param_name]

                    # Step 5: Calculate adjusted values: μ̂(xi) + w̄'γ̂
                    # This follows equation (2.6) in Cattaneo et al. (2023)
                    self.df.loc[group_mask, self.y_plot_var] = bin_effects + mean_cov_effect

                except Exception as e:
                    print(f"Warning: Error in covariate adjustment for group {group_val}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fall back to using original y values
                    self.df.loc[group_mask, self.y_plot_var] = self.df.loc[group_mask, self.y_var]

            # For any missing values, use the original
            missing_mask = self.df[self.y_plot_var].isna()
            if any(missing_mask):
                self.df.loc[missing_mask, self.y_plot_var] = self.df.loc[missing_mask, self.y_var]

    def _residualize(self, data, y_var, x_var, controls, fcontrols, weight_var=None):
        """Residualize y and x with respect to control variables."""
        data_res = data.copy()

        # Store original index info
        original_index = data_res.index
        original_index_name = original_index.name

        # Temporarily reset index for safe alignment
        data_res = data_res.reset_index()

        # Build control terms
        all_controls_terms = []
        if controls:
            if isinstance(controls, str):
                controls = [controls]
            all_controls_terms.extend(controls)
        if fcontrols:
            if isinstance(fcontrols, str):
                fcontrols = [fcontrols]
            fcontrol_str = " + ".join([f"C({f})" for f in fcontrols])
            all_controls_terms.append(fcontrol_str)

        # Check if controls actually exist in the dataframe
        valid_controls = [c for c in (controls or []) if c in data_res.columns]
        valid_fcontrols = [f for f in (fcontrols or []) if f in data_res.columns]

        all_valid_controls_terms = []
        if valid_controls:
            all_valid_controls_terms.extend(valid_controls)
        if valid_fcontrols:
            fcontrol_str = " + ".join([f"C({f})" for f in valid_fcontrols])
            all_valid_controls_terms.append(fcontrol_str)

        # Check if by_var is defined and in the data
        by_var = self.options['by_var']
        by_var_in_data = by_var is not None and by_var in data_res.columns

        # Early return if no valid controls
        if not all_valid_controls_terms:
            print("Warning: No valid control variables found. Skipping residualization.")

            # Add means back even if no controls, for consistency
            # If by_var is defined, compute and store group-specific means
            if by_var_in_data:
                # Initialize means columns
                data_res[f'{y_var}_resid'] = data_res[y_var]
                data_res[f'{x_var}_resid'] = data_res[x_var]

                # Calculate and store group-specific means
                for group_val in data_res[by_var].dropna().unique():
                    group_mask = data_res[by_var] == group_val

                    weights_for_mean = data_res.loc[
                        group_mask, weight_var] if weight_var and weight_var in data_res else None
                    weight_mask_y = weights_for_mean[
                        data_res.loc[group_mask, y_var].notna()] if weights_for_mean is not None else None
                    weight_mask_x = weights_for_mean[
                        data_res.loc[group_mask, x_var].notna()] if weights_for_mean is not None else None

                    y_mean = np.average(data_res.loc[group_mask, y_var].dropna(), weights=weight_mask_y)
                    x_mean = np.average(data_res.loc[group_mask, x_var].dropna(), weights=weight_mask_x)

                    # Store group-specific means in new columns
                    data_res.loc[group_mask, f'y_mean_{group_val}'] = y_mean
                    data_res.loc[group_mask, f'x_mean_{group_val}'] = x_mean
                    data_res.loc[group_mask, 'y_mean'] = y_mean  # Also store in the common column for compatibility
                    data_res.loc[group_mask, 'x_mean'] = x_mean
            else:
                # Original logic for overall means
                data_res[f'{y_var}_resid'] = data_res[y_var]
                data_res[f'{x_var}_resid'] = data_res[x_var]

                weights_for_mean = data_res[weight_var] if weight_var and weight_var in data_res else None
                weight_mask_y = weights_for_mean[data_res[y_var].notna()] if weights_for_mean is not None else None
                weight_mask_x = weights_for_mean[data_res[x_var].notna()] if weights_for_mean is not None else None

                data_res['y_mean'] = np.average(data_res[y_var].dropna(), weights=weight_mask_y)
                data_res['x_mean'] = np.average(data_res[x_var].dropna(), weights=weight_mask_x)

            # Restore original index
            self._restore_index(data_res, original_index_name)
            return data_res

        # Build formula for controls
        formula_controls_only = " + ".join(all_valid_controls_terms)
        weights = data_res[weight_var] if weight_var and weight_var in data_res else None

        # If by_var is defined, residualize and add means back by group
        if by_var_in_data:
            # Initialize residual columns
            data_res[f'{y_var}_resid'] = np.nan
            data_res[f'{x_var}_resid'] = np.nan

            # Process each group separately
            for group_val in data_res[by_var].dropna().unique():
                group_mask = data_res[by_var] == group_val
                group_data = data_res.loc[group_mask].copy()

                if group_data.empty:
                    continue

                # Calculate group-specific means
                group_weights_y = weights[group_data[y_var].notna()] if weights is not None else None
                y_mean_group = np.average(group_data[y_var].dropna(), weights=group_weights_y)

                group_weights_x = weights[group_data[x_var].notna()] if weights is not None else None
                x_mean_group = np.average(group_data[x_var].dropna(), weights=group_weights_x)

                # Store group-specific means
                data_res.loc[group_mask, f'y_mean_{group_val}'] = y_mean_group
                data_res.loc[group_mask, f'x_mean_{group_val}'] = x_mean_group
                data_res.loc[group_mask, 'y_mean'] = y_mean_group  # Also store in common column
                data_res.loc[group_mask, 'x_mean'] = x_mean_group

                # Residualize Y for this group
                self._residualize_single_variable(
                    group_data, y_var, formula_controls_only,
                    weights[group_mask] if weights is not None else None,
                    y_mean_group
                )

                # Residualize X for this group
                self._residualize_single_variable(
                    group_data, x_var, formula_controls_only,
                    weights[group_mask] if weights is not None else None,
                    x_mean_group
                )

                # Copy residuals back to the main dataframe
                data_res.loc[group_mask, f'{y_var}_resid'] = group_data[f'{y_var}_resid']
                data_res.loc[group_mask, f'{x_var}_resid'] = group_data[f'{x_var}_resid']
        else:
            # Calculate means of original variables for adding back later
            weights_y_mean = weights[data_res[y_var].notna()] if weights is not None else None
            y_mean_orig = np.average(data_res[y_var].dropna(), weights=weights_y_mean)

            weights_x_mean = weights[data_res[x_var].notna()] if weights is not None else None
            x_mean_orig = np.average(data_res[x_var].dropna(), weights=weights_x_mean)

            data_res['y_mean'] = y_mean_orig  # Store original mean
            data_res['x_mean'] = x_mean_orig

            # Residualize Y
            self._residualize_single_variable(
                data_res, y_var, formula_controls_only, weights, y_mean_orig
            )

            # Residualize X
            self._residualize_single_variable(
                data_res, x_var, formula_controls_only, weights, x_mean_orig
            )

        # Restore the original index
        self._restore_index(data_res, original_index_name)

        return data_res

    def _residualize_single_variable(self, data, var, formula_controls, weights, orig_mean):
        """Residualize a single variable against control variables."""
        # Create an explicit copy to avoid SettingWithCopyWarning
        data_copy = data.copy()
        formula = f"{var} ~ {formula_controls}"

        try:
            # Fit model
            if weights is not None:
                model = smf.wls(formula, data=data_copy, weights=weights, missing='drop').fit()
            else:
                model = smf.ols(formula, data=data_copy, missing='drop').fit()

            # Get residuals
            resid_series = model.resid

            # Assign back to data
            data.loc[:, f'{var}_resid'] = np.nan
            data.loc[resid_series.index, f'{var}_resid'] = resid_series + orig_mean

        except Exception as e:
            print(f"Warning: Could not residualize {var}. Error: {e}")
            # Fallback: copy original data
            data.loc[:, f'{var}_resid'] = data[var]

    def _setup_binning(self):
        """Setup binning for the data if binned option is True."""
        self.bin_col = None
        self.plot_binned_means = False

        if not self.options['binned']:
            return

        self.plot_binned_means = True
        self.bin_col = f"{self.x_var}_bin"

        try:
            if self.options['bin_var'] and self.options['bin_var'] in self.df:
                self.df[self.bin_col] = self.df[self.options['bin_var']]
            elif self.options['discrete']:
                # Use factorize for discrete values
                self.df[self.bin_col] = pd.factorize(self.df[self.x_plot_var])[0]
                self.df.loc[self.df[self.x_plot_var].isna(), self.bin_col] = np.nan  # Propagate NAs
            elif self.options['uni_bins'] is not None and self.options['uni_bins'] > 0:
                self.df[self.bin_col] = pd.cut(
                    self.df[self.x_plot_var],
                    bins=self.options['uni_bins'],
                    labels=False,
                    include_lowest=True
                )
            elif self.options['n_quantiles'] > 0:  # Default to quantiles
                # Handle potential non-unique edges in qcut
                try:
                    self.df[self.bin_col] = pd.qcut(
                        self.df[self.x_plot_var],
                        q=self.options['n_quantiles'],
                        labels=False,
                        duplicates='drop'
                    )
                except ValueError:  # If too few unique values for quantiles
                    print(f"Warning: Could not create {self.options['n_quantiles']} quantile bins. Using fewer bins.")
                    # Fallback: try discrete
                    self.df[self.bin_col] = pd.factorize(self.df[self.x_plot_var])[0]
                    self.df.loc[self.df[self.x_plot_var].isna(), self.bin_col] = np.nan
            else:  # No valid binning method specified
                self.plot_binned_means = False
                self.bin_col = None
                print("Warning: Invalid binning options specified. Plotting unbinned data.")

        except Exception as e:
            print(f"Warning: Could not create bins. Plotting unbinned data. Error: {e}")
            self.plot_binned_means = False
            self.bin_col = None

    def _residualize(self, data, y_var, x_var, controls, fcontrols, weight_var=None):
        """Residualize y and x with respect to control variables."""
        data_res = data.copy()

        # Store original index info
        original_index = data_res.index
        original_index_name = original_index.name

        # Temporarily reset index for safe alignment
        data_res = data_res.reset_index()

        # Build control terms
        all_controls_terms = []
        if controls:
            if isinstance(controls, str):
                controls = [controls]
            all_controls_terms.extend(controls)
        if fcontrols:
            if isinstance(fcontrols, str):
                fcontrols = [fcontrols]
            fcontrol_str = " + ".join([f"C({f})" for f in fcontrols])
            all_controls_terms.append(fcontrol_str)

        # Check if controls actually exist in the dataframe
        valid_controls = [c for c in (controls or []) if c in data_res.columns]
        valid_fcontrols = [f for f in (fcontrols or []) if f in data_res.columns]

        all_valid_controls_terms = []
        if valid_controls:
            all_valid_controls_terms.extend(valid_controls)
        if valid_fcontrols:
            fcontrol_str = " + ".join([f"C({f})" for f in valid_fcontrols])
            all_valid_controls_terms.append(fcontrol_str)

        # Check if by_var is defined and in the data
        by_var = self.options['by_var']
        by_var_in_data = by_var is not None and by_var in data_res.columns

        # Early return if no valid controls
        if not all_valid_controls_terms:
            print("Warning: No valid control variables found. Skipping residualization.")

            # Add means back even if no controls, for consistency
            # If by_var is defined, compute and store group-specific means
            if by_var_in_data:
                # Initialize means columns
                data_res[f'{y_var}_resid'] = data_res[y_var]
                data_res[f'{x_var}_resid'] = data_res[x_var]

                # Calculate and store group-specific means
                for group_val in data_res[by_var].dropna().unique():
                    group_mask = data_res[by_var] == group_val

                    weights_for_mean = data_res.loc[
                        group_mask, weight_var] if weight_var and weight_var in data_res else None
                    weight_mask_y = weights_for_mean[
                        data_res.loc[group_mask, y_var].notna()] if weights_for_mean is not None else None
                    weight_mask_x = weights_for_mean[
                        data_res.loc[group_mask, x_var].notna()] if weights_for_mean is not None else None

                    y_mean = np.average(data_res.loc[group_mask, y_var].dropna(), weights=weight_mask_y)
                    x_mean = np.average(data_res.loc[group_mask, x_var].dropna(), weights=weight_mask_x)

                    # Store group-specific means in new columns
                    data_res.loc[group_mask, f'y_mean_{group_val}'] = y_mean
                    data_res.loc[group_mask, f'x_mean_{group_val}'] = x_mean
                    data_res.loc[group_mask, 'y_mean'] = y_mean  # Also store in the common column for compatibility
                    data_res.loc[group_mask, 'x_mean'] = x_mean
            else:
                # Original logic for overall means
                data_res[f'{y_var}_resid'] = data_res[y_var]
                data_res[f'{x_var}_resid'] = data_res[x_var]

                weights_for_mean = data_res[weight_var] if weight_var and weight_var in data_res else None
                weight_mask_y = weights_for_mean[data_res[y_var].notna()] if weights_for_mean is not None else None
                weight_mask_x = weights_for_mean[data_res[x_var].notna()] if weights_for_mean is not None else None

                data_res['y_mean'] = np.average(data_res[y_var].dropna(), weights=weight_mask_y)
                data_res['x_mean'] = np.average(data_res[x_var].dropna(), weights=weight_mask_x)

            # Restore original index
            self._restore_index(data_res, original_index_name)
            return data_res

        # Build formula for controls
        formula_controls_only = " + ".join(all_valid_controls_terms)
        weights = data_res[weight_var] if weight_var and weight_var in data_res else None

        # If by_var is defined, residualize and add means back by group
        if by_var_in_data:
            # Initialize residual columns
            data_res[f'{y_var}_resid'] = np.nan
            data_res[f'{x_var}_resid'] = np.nan

            # Process each group separately
            for group_val in data_res[by_var].dropna().unique():
                group_mask = data_res[by_var] == group_val
                group_data = data_res.loc[group_mask].copy()

                if group_data.empty:
                    continue

                # Calculate group-specific means
                group_weights_y = weights[group_data[y_var].notna()] if weights is not None else None
                y_mean_group = np.average(group_data[y_var].dropna(), weights=group_weights_y)

                group_weights_x = weights[group_data[x_var].notna()] if weights is not None else None
                x_mean_group = np.average(group_data[x_var].dropna(), weights=group_weights_x)

                # Store group-specific means
                data_res.loc[group_mask, f'y_mean_{group_val}'] = y_mean_group
                data_res.loc[group_mask, f'x_mean_{group_val}'] = x_mean_group
                data_res.loc[group_mask, 'y_mean'] = y_mean_group  # Also store in common column
                data_res.loc[group_mask, 'x_mean'] = x_mean_group

                # Residualize Y for this group
                self._residualize_single_variable(
                    group_data, y_var, formula_controls_only,
                    weights[group_mask] if weights is not None else None,
                    y_mean_group
                )

                # Residualize X for this group
                self._residualize_single_variable(
                    group_data, x_var, formula_controls_only,
                    weights[group_mask] if weights is not None else None,
                    x_mean_group
                )

                # Copy residuals back to the main dataframe
                data_res.loc[group_mask, f'{y_var}_resid'] = group_data[f'{y_var}_resid']
                data_res.loc[group_mask, f'{x_var}_resid'] = group_data[f'{x_var}_resid']
        else:
            # Calculate means of original variables for adding back later
            weights_y_mean = weights[data_res[y_var].notna()] if weights is not None else None
            y_mean_orig = np.average(data_res[y_var].dropna(), weights=weights_y_mean)

            weights_x_mean = weights[data_res[x_var].notna()] if weights is not None else None
            x_mean_orig = np.average(data_res[x_var].dropna(), weights=weights_x_mean)

            data_res['y_mean'] = y_mean_orig  # Store original mean
            data_res['x_mean'] = x_mean_orig

            # Residualize Y
            self._residualize_single_variable(
                data_res, y_var, formula_controls_only, weights, y_mean_orig
            )

            # Residualize X
            self._residualize_single_variable(
                data_res, x_var, formula_controls_only, weights, x_mean_orig
            )

        # Restore the original index
        self._restore_index(data_res, original_index_name)

        return data_res

    def _residualize_single_variable(self, data, var, formula_controls, weights, orig_mean):
        """Residualize a single variable against control variables."""
        # Create an explicit copy to avoid SettingWithCopyWarning
        data_copy = data.copy()
        formula = f"{var} ~ {formula_controls}"

        try:
            # Fit model
            if weights is not None:
                model = smf.wls(formula, data=data_copy, weights=weights, missing='drop').fit()
            else:
                model = smf.ols(formula, data=data_copy, missing='drop').fit()

            # Get residuals
            resid_series = model.resid

            # Assign back to data
            data.loc[:, f'{var}_resid'] = np.nan
            data.loc[resid_series.index, f'{var}_resid'] = resid_series + orig_mean

        except Exception as e:
            print(f"Warning: Could not residualize {var}. Error: {e}")
            # Fallback: copy original data
            data.loc[:, f'{var}_resid'] = data[var]

    def _restore_index(self, data, original_index_name):
        """Restore the original index to the dataframe."""
        index_col = original_index_name if original_index_name else 'index'

        if index_col in data.columns:
            try:
                data.set_index(index_col, inplace=True)
                data.index.name = original_index_name
            except (KeyError, Exception) as e:
                print(f"Warning: Failed to restore original index. Error: {e}")

    def plot(self):
        """Create the scatter plot with fit line and return the axes."""
        # Set up the figure and axes
        self._setup_figure()

        # Determine colors for different groups
        scatter_color_map, fit_color_map = self._determine_colors()

        # Create the plot elements
        legend_handles = {}

        for group_val in self.by_groups:
            group_data = self._get_group_data(group_val)

            if group_data.empty:
                continue

            # Get colors for this group
            scatter_color = scatter_color_map[group_val]
            fit_color = fit_color_map[group_val]

            # Plot scatter points
            scatter_legend_label = self._plot_scatter_points(
                group_data,
                group_val,
                scatter_color,
                legend_handles
            )

            # Plot fit line if requested
            if self.options['fit'] != 'none':
                self._plot_fit_line(
                    group_data,
                    group_val,
                    fit_color,
                    legend_handles
                )

            # Add regression parameters if requested
            if self.options['regparameters']:
                self._add_reg_params(
                    group_data,
                    group_val,
                    fit_color
                )

        # Add KDE distribution of x variable if requested
        if self.options['xdistribution']:
            self._add_x_distribution(scatter_color_map)

        # Finalize the plot
        self._finalize_plot(legend_handles)

        return self.ax

    def _setup_figure(self):
        """Set up the figure and axes for plotting."""
        if self.options['ax'] is not None:
            self.ax = self.options['ax']
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.ax = ax

        # Style the plot
        self.ax.grid(True, linestyle='-', alpha=0.4, color='#B0B0B0', linewidth=1)  # Darker grid lines
        self.ax.set_facecolor('white')
        for spine in ['top', 'right']:
            self.ax.spines[spine].set_visible(False)
        DARK_GRAY = '#555555'
        for spine in ['bottom', 'left']:
            self.ax.spines[spine].set_color(DARK_GRAY)
            self.ax.spines[spine].set_linewidth(1.3)
        self.ax.tick_params(colors=DARK_GRAY, labelsize=11.5 * self.text_scale, width=1.5)

    def _determine_colors(self):
        """Determine colors for different groups."""
        scatter_color_map = {}
        fit_color_map = {}

        # Get unique by-groups
        if self.options['by_var'] and self.options['by_var'] in self.df:
            by_groups = sorted(self.df[self.options['by_var']].dropna().unique())
        else:
            by_groups = [None]  # Default for no grouping

        self.by_groups = by_groups

        # If grouping, determine color palette
        if self.options['by_var'] is not None and self.options['by_var'] in self.df:
            num_colors = len(by_groups)
            palette = self._get_color_palette(num_colors)
            for i, group_val in enumerate(by_groups):
                scatter_color_map[group_val] = palette[i]
                fit_color_map[group_val] = palette[i]  # Same color for scatter and fit per group

        else:  # No grouping
            scatter_color_map[None] = self.DEFAULT_SCATTER_COLOR
            fit_color_map[None] = self.DEFAULT_FIT_COLOR

        return scatter_color_map, fit_color_map

    def _get_color_palette(self, num_colors):
        """Get a color palette based on the colorscheme option."""
        if self.options['colorscheme'] is None:
            # Use default cycle
            palette_raw = [self.DEFAULT_COLOR_CYCLE[i % len(self.DEFAULT_COLOR_CYCLE)] for i in range(num_colors)]
            return [mcolors.to_rgba(c) for c in palette_raw]

        try:
            # Try getting a colormap
            cmap = plt.get_cmap(self.options['colorscheme'])
            palette = [cmap(i / max(1, num_colors - 1)) for i in range(num_colors)] if num_colors > 1 else [cmap(0.5)]

            if len(palette) < num_colors:  # Handle cases with fewer colors than needed
                palette = [cmap(i % cmap.N) for i in range(num_colors)]

            return palette

        except ValueError:
            # Try interpreting as a list of colors
            if isinstance(self.options['colorscheme'], list) and all(
                    isinstance(c, (str, tuple)) for c in self.options['colorscheme']):
                palette_raw = [self.options['colorscheme'][i % len(self.options['colorscheme'])] for i in
                               range(num_colors)]
                return [mcolors.to_rgba(c) for c in palette_raw]
            else:
                print(f"Warning: Invalid colorscheme '{self.options['colorscheme']}'. Using default color cycle.")

        except Exception as e:
            print(
                f"Warning: Error processing colorscheme '{self.options['colorscheme']}' ({e}). Using default color cycle.")

        # Fallback to default palette
        palette_raw = [self.DEFAULT_COLOR_CYCLE[i % len(self.DEFAULT_COLOR_CYCLE)] for i in range(num_colors)]
        return [mcolors.to_rgba(c) for c in palette_raw]

    def _get_group_data(self, group_val):
        """Get data for a specific group."""
        if self.options['by_var'] and self.options['by_var'] in self.df:
            if group_val is None:
                return pd.DataFrame()  # No data for None group if by_var is specified
            return self.df[self.df[self.options['by_var']] == group_val].copy()
        else:
            return self.df.copy()  # Return all data if no grouping

    def _get_binned_data(self, data, x_var, y_var, bin_col, weight_var=None):
        """Get binned data means for plotting."""
        if bin_col not in data or data[bin_col].isna().all():
            return pd.DataFrame()

        # Group by bin
        group_cols = [bin_col]

        # Group and aggregate
        try:
            # First calculate means for x and y
            agg_dict = {
                x_var: 'mean',
                y_var: 'mean'
            }

            # Perform aggregation
            binned_data = data.groupby(group_cols).agg(agg_dict).reset_index()

            # Add count column separately to avoid naming issues
            count_data = data.groupby(group_cols).size().reset_index(name='counts')
            binned_data = pd.merge(binned_data, count_data, on=group_cols)

            # Apply weights if available
            if weight_var in data:
                # Calculate weighted means
                weighted_means = data.groupby(group_cols).apply(
                    lambda g: pd.Series({
                        f"{x_var}_weighted": np.average(g[x_var], weights=g[weight_var]),
                        f"{y_var}_weighted": np.average(g[y_var], weights=g[weight_var]),
                    })
                ).reset_index()

                # Merge weighted means with main data
                binned_data = pd.merge(binned_data, weighted_means, on=group_cols)

                # Use weighted means
                binned_data[x_var] = binned_data[f"{x_var}_weighted"]
                binned_data[y_var] = binned_data[f"{y_var}_weighted"]

            return binned_data

        except Exception as e:
            print(f"Warning: Error in binning. Error: {e}")
            return pd.DataFrame()

    def _plot_scatter_points(self, group_data, group_val, scatter_color, legend_handles):
        """Plot scatter points for a group."""
        group_label = str(group_val) if group_val is not None else None
        scatter_legend_label = group_label if group_label else 'Data'

        # Extract the base data
        x_data = group_data[self.x_plot_var]
        y_data = group_data[self.y_plot_var]
        scatter_x = x_data
        scatter_y = y_data

        # Marker size
        if self.marker_size is not None:
            base_marker_size = plt.rcParams['lines.markersize'] ** 2 *.9 * self.marker_size
        else:
            base_marker_size = plt.rcParams['lines.markersize'] ** 2 *.9
        point_count = len(scatter_x.dropna())
        scatter_s = base_marker_size * max(0.25, min(2.5, 1.4 * np.sqrt(400 / max(1, point_count))))

        # Process binned data if needed
        if self.plot_binned_means and self.bin_col is not None and self.bin_col in group_data:
            binned_group_data = self._get_binned_data(
                group_data,
                self.x_plot_var,
                self.y_plot_var,
                self.bin_col,
                self.options['weight_var']
            )

            if not binned_group_data.empty:
                scatter_x = binned_group_data[self.x_plot_var]
                scatter_y = binned_group_data[self.y_plot_var]
                scatter_legend_label = f"{group_label} (Binned)" if group_label else "Binned Means"

                # Adjust size for binned data - larger markers since fewer points
                bin_count = len(scatter_x.dropna())
                scatter_s = base_marker_size * max(0.25, min(2.5, 1.4 * np.sqrt(400 / max(1, bin_count))))

                # Weight markers by bin count if requested
                if self.options['mweighted'] and 'counts' in binned_group_data:
                    mean_count = binned_group_data['counts'].replace(0, np.nan).mean()

                    if mean_count > 0:
                        scatter_s = scatter_s * (binned_group_data['counts'] / mean_count) * 1.5
                        scatter_s = np.clip(scatter_s, scatter_s * 0.2, scatter_s * 5)
                    else:
                        scatter_s = scatter_s

        # Apply jitter if requested (only for non-binned data)
        if self.options['jitter'] and not self.plot_binned_means:
            x_no_nan = scatter_x.dropna()
            y_no_nan = scatter_y.dropna()

            if len(x_no_nan) > 1 and len(y_no_nan) > 1:
                x_range = x_no_nan.max() - x_no_nan.min()
                y_range = y_no_nan.max() - y_no_nan.min()

                if x_range > 1e-9:
                    x_jitter = np.random.uniform(-self.options['jitter'] / 2, self.options['jitter'] / 2,
                                                 size=len(scatter_x)) * x_range
                    scatter_x = scatter_x.add(pd.Series(x_jitter, index=scatter_x.index), fill_value=0)

                if y_range > 1e-9:
                    y_jitter = np.random.uniform(-self.options['jitter'] / 2, self.options['jitter'] / 2,
                                                 size=len(scatter_y)) * y_range
                    scatter_y = scatter_y.add(pd.Series(y_jitter, index=scatter_y.index), fill_value=0)

        # Filter valid data points
        valid_indices = scatter_x.notna() & scatter_y.notna()
        if not valid_indices.any():
            return scatter_legend_label

        # Handle marker labels if requested
        if self.options['mlabel'] and self.options['mlabel'] in group_data.columns:
            # Using text labels instead of markers
            handle = self.ax.scatter(
                [], [],
                color=scatter_color,
                label=scatter_legend_label,
                s=np.mean(scatter_s) if isinstance(scatter_s, (pd.Series, np.ndarray)) else scatter_s
            )

            legend_handles[scatter_legend_label] = handle

            # Plot text labels at scatter points
            for i, (x, y) in enumerate(zip(scatter_x[valid_indices], scatter_y[valid_indices])):
                label_value = group_data.loc[valid_indices, self.options['mlabel']].iloc[i]
                self.ax.text(
                    x, y, str(label_value),
                    color=scatter_color,
                    fontsize=8,
                    ha='center', va='center'
                )
        else:
            # Regular scatter plot
            handle = self.ax.scatter(
                scatter_x[valid_indices], scatter_y[valid_indices],
                color=scatter_color,
                label=scatter_legend_label,
                s=scatter_s[valid_indices] if isinstance(scatter_s, pd.Series) else scatter_s,
                alpha=0.5,
                edgecolors=mcolors.to_rgba(scatter_color, alpha=0.9),
                linewidths=1.6
            )

            legend_handles[scatter_legend_label] = handle

        return scatter_legend_label

    def _plot_fit_line(self, group_data, group_val, fit_color, legend_handles):
        """Plot fit line for a group."""
        group_label = str(group_val) if group_val is not None else None

        # Get options
        fit_type = self.options['fit']
        ci = self.options['ci']
        level = self.options['level']
        bw_frac = self.options['bw_frac']
        binary_model = self.options['fitmodel']

        # For binary DV with controls, handle special case
        residualized = self.y_plot_var.endswith('_resid') and self.x_plot_var.endswith('_resid')

        if self.binary_dv and residualized and (self.options['controls'] or self.options['fcontrols']):
            # Get original variable names by removing '_resid' suffix
            orig_y_var = self.y_var
            orig_x_var = self.x_var

            # Create a copy with both residualized and original variables
            df_plot = group_data.dropna(subset=[self.y_plot_var, self.x_plot_var, orig_y_var, orig_x_var]).copy()

            # Use original variables for fit, but keep using residualized vars for plotting range
            fit_y_var = orig_y_var
            fit_x_var = orig_x_var
        else:
            # Standard case - use the provided variables
            df_plot = group_data.dropna(subset=[self.y_plot_var, self.x_plot_var]).copy()
            fit_y_var = self.y_plot_var
            fit_x_var = self.x_plot_var

        if df_plot.empty:
            return False

        # Ensure y_var is numeric for binary models
        if self.binary_dv and df_plot[fit_y_var].dtype.name in ['category', 'object']:
            try:
                # Try to convert categorical to numeric
                if df_plot[fit_y_var].dtype.name == 'category':
                    # For binary categorical, map to 0/1
                    if len(df_plot[fit_y_var].cat.categories) == 2:
                        df_plot[fit_y_var] = df_plot[fit_y_var].cat.codes.astype(float)
                    else:
                        df_plot[fit_y_var] = pd.to_numeric(df_plot[fit_y_var], errors='coerce')
                else:
                    # For string/object type with binary values
                    unique_vals = sorted(df_plot[fit_y_var].astype(str).unique())
                    if len(unique_vals) == 2:
                        # Map lower value to 0, higher to 1
                        df_plot[fit_y_var] = df_plot[fit_y_var].astype(str).map(
                            {unique_vals[0]: 0, unique_vals[1]: 1}).astype(float)
                    else:
                        # Try direct conversion
                        df_plot[fit_y_var] = pd.to_numeric(df_plot[fit_y_var], errors='coerce')
            except Exception as e:
                print(f"Warning: Could not convert binary y_var to numeric: {e}")
                return False

        # Create prediction range based on the x variable
        x_pred_range = np.linspace(df_plot[self.x_plot_var].min(), df_plot[self.x_plot_var].max(), 100)
        exog_pred = pd.DataFrame({fit_x_var: x_pred_range})  # Use fit_x_var for prediction

        # Add mean of controls if they were used (needed for prediction)
        controls = self.options['controls'] or []
        fcontrols = self.options['fcontrols'] or []
        all_control_vars = controls + fcontrols

        if all_control_vars:
            # Add means of continuous controls
            for ctrl in controls:
                if ctrl in df_plot:
                    ctrl_mean = df_plot[ctrl].mean()
                    exog_pred[ctrl] = ctrl_mean

            # For factor controls use most common category
            for fctrl in fcontrols:
                if fctrl in df_plot:
                    if df_plot[fctrl].dtype.name == 'category':
                        most_common = df_plot[fctrl].value_counts().idxmax()
                        exog_pred[fctrl] = most_common
                    else:
                        # If not category, treat as continuous and use mean
                        exog_pred[fctrl] = df_plot[fctrl].mean()

        # Set up weights if available
        weights = df_plot[self.options['weight_var']] if self.options['weight_var'] in df_plot else None
        alpha = 1 - level / 100.0

        # Fit model and plot
        fit_successful = self._fit_and_plot_model(
            df_plot, fit_y_var, fit_x_var, x_pred_range, exog_pred,
            fit_type, ci, alpha, bw_frac,
            binary_model, weights, fit_color
        )

        # Add to legend if successful
        if fit_successful:
            fit_label = f"{group_label} ({fit_type} fit)" if group_label else f"{fit_type.capitalize()} fit"
            dummy_line = plt.Line2D([0], [0], color=fit_color, lw=2.5, label=fit_label)
            legend_handles[fit_label] = dummy_line

            if ci:
                ci_label = f"{level:.0f}% CI"
                if ci_label not in legend_handles:
                    dummy_fill = plt.Rectangle((0, 0), 1, 1, fc=mcolors.to_rgba(fit_color, alpha=0.2), label=ci_label)
                    legend_handles[ci_label] = dummy_fill

        return fit_successful

    def _fit_and_plot_model(self, df_plot, fit_y_var, fit_x_var, x_pred_range, exog_pred,
                            fit_type, ci, alpha, bw_frac, binary_model, weights, fit_color):
        """Fit statistical model and plot the fit line with confidence intervals."""
        try:
            # --- LOWESS / LPOLY ---
            if fit_type in ['lowess', 'lpoly']:
                if len(df_plot) < 3:  # Need enough points for lowess
                    print(f"Warning: Too few points ({len(df_plot)}) for LOWESS/Lpoly fit. Skipping.")
                    return False

                # Use statsmodels lowess
                smoothed = lowess(
                    df_plot[fit_y_var], df_plot[fit_x_var],
                    frac=bw_frac, it=0, is_sorted=False,
                    delta=0.0,  # Use local regression, not distance weighting
                    return_sorted=True  # Ensure result is sorted on X for plotting
                )

                if smoothed is not None and len(smoothed) > 0:
                    self.ax.plot(smoothed[:, 0], smoothed[:, 1], color=fit_color, lw=2.5, label='_nolegend_')
                    if ci:
                        print("Warning: Confidence intervals for LOWESS/Lpoly not supported. Omitted.")
                    return True

            # --- Parametric Models ---
            else:
                formula = f"{fit_y_var} ~ "
                if fit_type == 'linear':
                    formula += f"{fit_x_var}"
                elif fit_type == 'quadratic':
                    formula += f"np.power({fit_x_var}, 1) + np.power({fit_x_var}, 2)"
                elif fit_type == 'cubic':
                    formula += f"np.power({fit_x_var}, 1) + np.power({fit_x_var}, 2) + np.power({fit_x_var}, 3)"
                else:
                    print(f"Warning: Unknown fit type '{fit_type}'. Using linear.")
                    formula += f"{fit_x_var}"

                # Add controls to formula when using original variables
                controls = self.options['controls'] or []
                fcontrols = self.options['fcontrols'] or []

                if controls or fcontrols:
                    # Add continuous controls
                    if controls:
                        control_terms = [c for c in controls if c in df_plot.columns]
                        if control_terms:
                            formula += " + " + " + ".join(control_terms)

                    # Add categorical controls
                    if fcontrols:
                        fcontrol_terms = [f"C({f})" for f in fcontrols if f in df_plot.columns]
                        if fcontrol_terms:
                            formula += " + " + " + ".join(fcontrol_terms)

                # Select model type
                model_base = smf.ols  # Default
                model_args = {'formula': formula, 'data': df_plot}

                if weights is not None:
                    model_args['weights'] = weights
                    model_base = smf.wls

                if self.binary_dv:
                    if binary_model == 'logit':
                        model_base = smf.logit
                    elif binary_model == 'probit':
                        model_base = smf.probit
                    elif binary_model == 'lpm':
                        pass  # Already WLS/OLS
                    else:
                        model_base = smf.logit  # Default binary to logit

                # Fit model
                model = model_base(**model_args).fit(disp=False)

                # Prepare prediction data for exog_pred
                # Add controls to the prediction dataset
                for ctrl in controls:
                    if ctrl in df_plot.columns and ctrl not in exog_pred:
                        exog_pred[ctrl] = df_plot[ctrl].mean()

                for fctrl in fcontrols:
                    if fctrl in df_plot.columns:
                        # Get unique values as they appear in the training data
                        unique_values = df_plot[fctrl].dropna().unique()
                        if len(unique_values) > 0:
                            # Use most common value from original data
                            most_common = df_plot[fctrl].value_counts().idxmax()

                            # Ensure we're using a value that was in the original data
                            if fctrl in exog_pred:
                                current_value = exog_pred[fctrl].iloc[0] if hasattr(exog_pred[fctrl], 'iloc') else \
                                exog_pred[fctrl]
                                # Check if current value is not in the original categories
                                if current_value not in unique_values:
                                    # Replace with a valid value from the training data
                                    exog_pred[fctrl] = most_common
                            else:
                                # If not in exog_pred, add it with the most common value
                                exog_pred[fctrl] = most_common

                # Get predictions
                try:
                    # For binary models, ensure exog_pred matches the model's expected format
                    if hasattr(model, 'model') and hasattr(model.model, 'data'):
                        design_info = model.model.data.design_info
                        if design_info is not None:
                            from patsy import dmatrix
                            try:
                                # Try direct prediction first as a fallback
                                pred = model.get_prediction(exog_pred)
                            except Exception as e:
                                # If that fails, don't try design matrix again - use manual approach
                                print(f"Warning: Error in prediction: {e}")
                                print("Attempting alternative prediction method...")

                                # Create prediction values manually
                                x_values = exog_pred[fit_x_var].values
                                y_pred_values = []

                                # Get coefficients from the model
                                params = model.params
                                intercept = params.get('Intercept', 0)
                                slope = params.get(fit_x_var, 0)

                                # Simple prediction for linear model
                                if fit_type == 'linear':
                                    y_pred_values = intercept + slope * x_values
                                else:
                                    # Fallback for other models - use linear approximation
                                    print("Warning: Using linear approximation for fit line.")
                                    y_pred_values = intercept + slope * x_values

                                # Create a DataFrame similar to pred.summary_frame()
                                import pandas as pd
                                pred_df = pd.DataFrame({
                                    'mean': y_pred_values,
                                })

                                # Skip confidence intervals in manual prediction
                                ci_lower = ci_upper = None

                                # Plot the fit line
                                self.ax.plot(x_pred_range, y_pred_values, color=fit_color, lw=2.5, label='_nolegend_')
                                return True
                        else:
                            pred = model.get_prediction(exog_pred)
                    else:
                        pred = model.get_prediction(exog_pred)

                    pred_df = pred.summary_frame(alpha=alpha)

                    # Handle different column name formats from statsmodels
                    if 'mean' in pred_df.columns:
                        y_pred = pred_df['mean']
                        ci_lower = pred_df.get('mean_ci_lower', None)
                        ci_upper = pred_df.get('mean_ci_upper', None)
                    elif 'predicted' in pred_df.columns:
                        y_pred = pred_df['predicted']
                        ci_lower = pred_df.get('ci_lower', None)
                        ci_upper = pred_df.get('ci_upper', None)
                    elif 'predicted_mean' in pred_df.columns:
                        y_pred = pred_df['predicted_mean']
                        ci_lower = pred_df.get('obs_ci_lower', None)
                        ci_upper = pred_df.get('obs_ci_upper', None)
                    else:
                        # Fallback to first column
                        print(f"Warning: Unexpected prediction columns: {pred_df.columns}")
                        y_pred = pred_df.iloc[:, 0]  # Use first column as prediction
                        ci_lower = ci_upper = None

                    # Plot the fit line
                    self.ax.plot(x_pred_range, y_pred, color=fit_color, lw=2.5, label='_nolegend_')

                    # Add confidence intervals if requested
                    if ci and ci_lower is not None and ci_upper is not None:
                        self.ax.fill_between(
                            x_pred_range, ci_lower, ci_upper,
                            color=fit_color, alpha=0.2, label='_nolegend_'
                        )

                    return True

                except Exception as pred_e:
                    print(f"Warning: Could not get predictions for fit line. Error: {pred_e}")
                    import traceback
                    traceback.print_exc()

                    # Fallback: try to plot a simplified fit line
                    try:
                        # Get coefficients directly
                        intercept = model.params.get('Intercept', 0)
                        slope = model.params.get(fit_x_var, 0)

                        # Plot a simple linear fit even if the original model was more complex
                        y_pred_simple = intercept + slope * x_pred_range
                        self.ax.plot(x_pred_range, y_pred_simple, color=fit_color, lw=2.5,
                                     linestyle='--', label='_nolegend_')

                        print("Warning: Using simplified linear approximation for fit line.")
                        return True
                    except Exception as fallback_e:
                        print(f"Warning: Fallback plotting also failed: {fallback_e}")

        except Exception as e:
            print(f"Warning: Could not fit model type '{fit_type}'. Error: {e}")
            import traceback
            traceback.print_exc()

        return False

    def _add_reg_params(self, group_data, group_val, color):
        """Add regression parameters to the plot."""
        if not self.options['regparameters']:
            return

        regparameters = self.options['regparameters']
        by_var = self.options['by_var']
        bymethod = self.options['bymethod']

        # Filter data
        df_reg = group_data.dropna(
            subset=[self.y_plot_var, self.x_plot_var] +
                   (self.options['controls'] or []) +
                   (self.options['fcontrols'] or [])
        ).copy()

        if df_reg.empty:
            return

        # Setup weights
        weights = df_reg[self.options['weight_var']] if self.options['weight_var'] in df_reg else None

        # --- Build Formula ---
        formula = f"{self.y_plot_var} ~ {self.x_plot_var}"  # Base linear relationship

        # Handle interaction method
        if by_var and bymethod == 'interact':
            formula += f" * C({by_var})"

        # Add controls
        if self.options['controls']:
            formula += " + " + " + ".join(self.options['controls'])
        if self.options['fcontrols']:
            formula += " + " + " + ".join([f"C({f})" for f in self.options['fcontrols']])

        # --- Fit Model ---
        try:
            model_base = smf.ols
            model_args = {'formula': formula, 'data': df_reg}

            if weights is not None:
                model_args['weights'] = weights
                model_base = smf.wls

            model = model_base(**model_args).fit(disp=False)
            results_summary = model.summary2().tables[1]
            nobs = int(model.nobs)
            rsquared = model.rsquared
            rsquared_adj = model.rsquared_adj

        except Exception as e:
            print(f"Warning: Could not fit regression for parameters. Error: {e}")
            return

        # --- Extract Parameters ---
        param_text_lines = []  # Build lines of text
        target_x_param = self.x_plot_var
        target_int_params = []

        # Determine which coefficient/interaction to extract based on bymethod
        if by_var:
            if bymethod == 'stratify':
                # Refit on the subset for stratified parameters
                try:
                    df_subset = df_reg[df_reg[by_var] == group_val]
                    if df_subset.empty:
                        return  # Skip if subset is empty

                    # Build formula without interaction for subset
                    formula_strat = f"{self.y_plot_var} ~ {self.x_plot_var}"

                    if self.options['controls']:
                        formula_strat += " + " + " + ".join(self.options['controls'])
                    if self.options['fcontrols']:
                        formula_strat += " + " + " + ".join([f"C({f})" for f in self.options['fcontrols']])

                    model_args_strat = {'formula': formula_strat, 'data': df_subset}

                    if weights is not None:
                        subset_weights = df_subset[self.options['weight_var']].dropna()
                        if subset_weights.empty and len(df_subset) > 0:
                            print(f"Warning: No valid weights for group {group_val}, using unweighted params.")
                            model_base_strat = smf.ols
                        else:
                            model_args_strat['weights'] = subset_weights
                            model_base_strat = smf.wls
                    else:
                        model_base_strat = smf.ols

                    model_strat = model_base_strat(**model_args_strat).fit()
                    results_summary = model_strat.summary2().tables[1]
                    nobs = int(model_strat.nobs)
                    rsquared = model_strat.rsquared
                    rsquared_adj = model_strat.rsquared_adj
                    target_x_param = self.x_plot_var  # Simple slope from subset model

                except Exception as e:
                    print(f"Warning: Could not fit stratified regression for group {group_val}. Error: {e}")
                    return  # Skip params for this group

            elif bymethod == 'interact':
                # Extract marginal effect for this group and interaction terms
                try:
                    base_x_row = results_summary.loc[self.x_plot_var]
                    base_x_coef = base_x_row['Coef.']

                    # Find reference level (first unique value encountered by patsy)
                    ref_level = \
                    model.model.data.frame[by_var].iloc[model.model.data.row_labels].astype('category').cat.categories[
                        0]

                    if group_val == ref_level:  # Current group is the reference level
                        current_x_coef = base_x_coef
                        current_x_pval = base_x_row['P>|t|']  # P-value of base slope
                    else:
                        # Find interaction term coefficient for this specific level
                        interaction_term = f"{self.x_plot_var}:C({by_var})[T.{group_val}]"
                        if interaction_term in results_summary.index:
                            int_row = results_summary.loc[interaction_term]
                            int_coef = int_row['Coef.']
                            current_x_coef = base_x_coef + int_coef
                            current_x_pval = np.nan  # P-value requires delta method
                            target_int_params.append({
                                'name': interaction_term,
                                'coef': int_coef,
                                'pval': int_row['P>|t|']
                            })
                        else:
                            print(f"Warning: Could not find interaction term {interaction_term}.")
                            current_x_coef = np.nan
                            current_x_pval = np.nan

                except (KeyError, AttributeError, IndexError) as e:
                    print(f"Warning: Could not extract interaction parameters for group {group_val}. Error: {e}")
                    return  # Skip params for this group

        # --- Build Text String ---
        param_map = {'coef': 'Coef.', 'se': 'Std.Err.', 'pval': 'P>|t|'}
        sig_levels = {0.01: '***', 0.05: '**', 0.1: '*'}

        # Add parameters based on 'regparameters' list
        line_content = []
        group_prefix = f"{group_val}: " if by_var else ""

        # Main Slope Coefficient
        if 'coef' in regparameters or 'sig' in regparameters:
            coef_val = np.nan
            pval_val = np.nan
            se_val = np.nan

            if by_var and bymethod == 'interact':
                coef_val = current_x_coef
                pval_val = current_x_pval  # This is approximate (base or nan)
                # SE not easily available without margins
            elif target_x_param in results_summary.index:
                row = results_summary.loc[target_x_param]
                coef_val = row[param_map['coef']]
                pval_val = row[param_map['pval']]
                se_val = row[param_map['se']]

            if not pd.isna(coef_val):
                sig_star = ''
                if not pd.isna(pval_val):
                    for level, stars in sig_levels.items():
                        if pval_val < level:
                            sig_star = stars
                            break
                if 'coef' in regparameters:
                    line_content.append(f"β={coef_val:.3f}{sig_star}")
                elif 'sig' in regparameters:
                    line_content.append(sig_star)  # Just add stars if coef not requested

            if 'se' in regparameters and not pd.isna(se_val):
                line_content.append(f"(SE={se_val:.3f})")
            if 'pval' in regparameters and not pd.isna(pval_val):
                # Avoid showing approximate p-value for interaction marginal effect
                if not (by_var and bymethod == 'interact' and group_val != ref_level):
                    line_content.append(f"(p={pval_val:.3f})")

        if line_content:
            param_text_lines.append(group_prefix + " ".join(line_content))

        # Interaction Terms (only if bymethod='interact')
        if 'int' in regparameters and by_var and bymethod == 'interact':
            for int_p in target_int_params:
                int_line = []
                int_coef = int_p['coef']
                int_pval = int_p['pval']
                sig_star = ''
                for level, stars in sig_levels.items():
                    if int_pval < level:
                        sig_star = stars
                        break

                # Extract interacting level name cleanly
                try:
                    level_name = int_p['name'].split(':')[-1].split('[T.')[-1].replace(']', '')
                except:
                    level_name = 'Int'

                int_line.append(f"β_int({level_name})={int_coef:.3f}{sig_star}")
                if 'pval' in regparameters:
                    int_line.append(f"(p={int_pval:.3f})")

                param_text_lines.append("  " + " ".join(int_line))  # Indent interaction term

        # R-squared and N (show only once if interact, or per group if stratify)
        show_stats = not (by_var and bymethod == 'interact' and group_val != ref_level) or not by_var
        if show_stats:
            stats_line = []
            if 'r2' in regparameters:
                stats_line.append(f"R²={rsquared:.3f}")
            if 'adjr2' in regparameters:
                stats_line.append(f"Adj.R²={rsquared_adj:.3f}")
            if 'nobs' in regparameters:
                stats_line.append(f"N={nobs}")

            if stats_line:
                param_text_lines.append(group_prefix + " ".join(stats_line))

        # Combine lines into final text
        full_text = "\n".join(param_text_lines)

        # --- Add to Plot ---
        if full_text:
            # Position calculation
            parpos = self.options['parpos']
            if parpos:
                try:
                    pos = [float(p) for p in parpos.split()]
                    x_pos, y_pos = pos[0], pos[1]
                    ha, va = 'left', 'bottom'
                    if x_pos > 0.5:
                        ha = 'right'
                    if y_pos > 0.5:
                        va = 'top'
                except:
                    print("Warning: Invalid parpos format. Using auto position.")
                    x_pos, y_pos, ha, va = 0.98, 0.98, 'right', 'top'
            else:
                x_pos, y_pos, ha, va = 0.98, 0.98, 'right', 'top'  # Default: top-right

            # Color code stratified text
            text_color = color if (by_var and bymethod == 'stratify') else self.DEFAULT_TEXT_COLOR

            # Add text
            self.ax.text(
                x_pos, y_pos, full_text,
                transform=self.ax.transAxes,
                fontsize=self.options['parsize'] if self.options['parsize'] else 10 * self.text_scale,
                ha=ha, va=va, color=text_color, horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.7, ec='#555555', lw=1)
            )

    def _add_x_distribution(self, scatter_color_map):
        """Add kernel density plot of x variable distribution."""
        if not self.options['xdistribution']:
            return

        # Parse x distribution options
        kde_height_ratio = 0.15
        kde_pos_req = 'auto'  # Default

        dist_parts = self.options['xdistribution'].split()
        if len(dist_parts) > 0:
            kde_pos_req = dist_parts[0].lower()
        if len(dist_parts) > 1:
            try:
                kde_height_ratio = float(dist_parts[1])
            except ValueError:
                print("Warning: Invalid height ratio for xdistribution. Using default.")

        # Validate position
        if kde_pos_req not in ['top', 'bottom', 'auto']:
            print("Warning: Invalid xdistribution position. Use 'top', 'bottom', or 'auto'. Defaulting to 'top'.")
            kde_pos = 'top'
        elif kde_pos_req == 'auto':
            kde_pos = 'top'
        else:
            kde_pos = kde_pos_req

        try:
            # Create the KDE axes
            divider = make_axes_locatable(self.ax)

            if kde_pos == 'top':
                kde_ax = divider.append_axes("top", size=f"{kde_height_ratio * 100}%", pad=0.05, sharex=self.ax)
                kde_ax.xaxis.set_tick_params(labelbottom=False)
            elif kde_pos == 'bottom':
                kde_ax = divider.append_axes("bottom", size=f"{kde_height_ratio * 100}%", pad=0.05, sharex=self.ax)
                self.ax.xaxis.set_tick_params(labelbottom=False)

            # Style the KDE axes
            kde_ax.yaxis.set_major_locator(mticker.NullLocator())
            kde_ax.tick_params(axis='x', colors=self.DEFAULT_TEXT_COLOR, labelsize=8)
            kde_ax.set_facecolor('white')

            for spine in ['top', 'right', 'left']:
                kde_ax.spines[spine].set_visible(False)

            kde_ax.spines['bottom'].set_color(self.DEFAULT_EDGE_COLOR)

            if kde_pos == 'bottom':
                self.ax.spines['bottom'].set_visible(False)

            # Plot KDE for each group or overall
            for group_val in self.by_groups:
                kde_color = scatter_color_map[group_val]
                group_data = self._get_group_data(group_val)

                if not group_data.empty:
                    x_kde_data = group_data[self.x_plot_var].dropna()

                    if len(x_kde_data) > 1:
                        try:
                            # Create KDE
                            kde = gaussian_kde(x_kde_data, bw_method=self.options['xdistrbw'])
                            x_range = np.linspace(x_kde_data.min(), x_kde_data.max(), 200)
                            kde_y = kde(x_range)

                            # Plot KDE line and fill
                            kde_ax.plot(x_range, kde_y, color=kde_color, lw=1.5)
                            kde_ax.fill_between(x_range, kde_y, color=kde_color, alpha=0.3)

                        except Exception as e:
                            print(f"Warning: Could not compute KDE for group {group_val}. Error: {e}")

            # Add density label
            kde_ax.set_ylabel('Density', fontsize=8, color=self.DEFAULT_TEXT_COLOR)
            kde_ax.yaxis.set_label_coords(-0.01, 0.5)

            # Store the KDE axis
            self.kde_ax = kde_ax

        except Exception as e:
            print(f"Error setting up KDE axes: {e}")
            self.kde_ax = None  # Reset if setup fails

    def _finalize_plot(self, legend_handles):
        """Add legend, titles, and finalize the plot."""
        # Add legend if we have handles
        if legend_handles:
            self.ax.legend(
                handles=legend_handles.values(),
                labels=legend_handles.keys(),
                fontsize=10,
                frameon=True,
                facecolor='white',
                edgecolor=self.DEFAULT_EDGE_COLOR,
                borderpad=0.8,
                loc='best'
            )

        # Set axis labels with larger font sizes
        final_xlabel = self.options['xtitle'] if self.options['xtitle'] is not None else self.x_var
        final_ylabel = self.options['ytitle'] if self.options['ytitle'] is not None else self.y_var

        # Apply labels to the correct axes, considering KDE plot position
        if self.kde_ax and hasattr(self.options['xdistribution'], 'split') and self.options['xdistribution'].split()[
            0].lower() == 'bottom':
            # If KDE is on bottom, KDE axis gets the x label
            self.kde_ax.set_xlabel(final_xlabel, color=self.DEFAULT_TEXT_COLOR, size=14* self.text_scale)
            self.ax.set_xlabel("")
            self.ax.set_ylabel(final_ylabel, color=self.DEFAULT_TEXT_COLOR, size=14* self.text_scale)
        else:
            # Default case: labels on the main axis
            self.ax.set_xlabel(final_xlabel, color=self.DEFAULT_TEXT_COLOR, size=14* self.text_scale)
            self.ax.set_ylabel(final_ylabel, color=self.DEFAULT_TEXT_COLOR, size=14* self.text_scale)

        # Set title color if exists
        if self.ax.get_title():
            self.ax.title.set_color('black')
            self.ax.title.set_size(16* self.text_scale)
            self.ax.title.set_fontweight('bold')

        # Adjust layout
        try:
            plt.tight_layout()
        except ValueError:
            print("Warning: tight_layout failed. Plot may have overlapping elements.")


# Create a convenience wrapper function
def scatterfit(data, y_var, x_var, return_ax=False, text_scale=1.0, marker_size=None, **kwargs):
    """
    Generates scatter plots with fit lines, similar to Stata's scatterfit.
    A wrapper around the ScatterFit class for backward compatibility.

    Args:
        data: Pandas DataFrame containing the data.
        y_var: Name of the dependent variable column.
        x_var: Name of the independent variable column.
        return_ax: If True, returns the matplotlib Axes object (default: False).
        **kwargs: Optional parameters (see ScatterFit class for details).

    Returns:
        matplotlib.axes.Axes: The Axes object containing the plot if return_ax is True, otherwise None.
    """
    scatter_fit = ScatterFit(data, y_var, x_var, text_scale=text_scale, marker_size=marker_size, **kwargs)
    ax = scatter_fit.plot()
    if return_ax:
        return ax
    return None