"""
grade_progression.py

This file defines the mathematical expectations, number ranges, and skill levels
for grades 1 through 8. It acts as the *knowledge backbone* for the entire agent.

"""

GRADE_PROGRESSION = {
    1: {
        "description": "Grade 1 builds fluent counting, +/- within 20, and reasoning with shapes.",
        "reference": "CA CCSSM Grade 1 (2023 framework)",
        "numbers": {
            "range": (0, 120),
            "place_value": True,
            "skip_counting": [2, 5, 10],
        },
        "operations": [
            "add_within_20_fluency",
            "sub_within_20_fluency",
            "make_ten_strategy",
            "compare_two_numbers",
        ],
        "fractions": [
            "partition_shapes_halves_quarters",
            "equal_shares_language",
        ],
        "geometry": [
            "compose_decompose_2d_shapes",
            "recognize_3d_shapes",
            "equal_halves_quarters",
        ],
        "measurement": [
            "length_nonstandard_units",
            "time_hours_half_hours",
            "compare_lengths",
        ],
        "data": [
            "picture_graphs_within_20",
            "bar_graphs_single_unit",
        ],
        "algebraic_thinking": [
            "counting_patterns",
            "true_false_equations",
            "unknown_addend_models",
        ],
    },

    2: {
        "description": "Grade 2 extends place value to 1,000, fluency with +/- within 100, and arrays.",
        "reference": "CA CCSSM Grade 2 (2023 framework)",
        "numbers": {
            "range": (0, 1000),
            "place_value": True,
            "skip_counting": [2, 3, 4, 5, 10],
        },
        "operations": [
            "add_sub_within_100",
            "add_sub_with_regrouping",
            "equal_groups_foundations",
            "array_models_for_multiplication",
        ],
        "fractions": [
            "partition_rectangles_equal_shares",
            "label_halves_thirds_fourths",
            "unit_fraction_language",
        ],
        "geometry": [
            "classify_polygons_by_sides",
            "partition_rectangles_rows_columns",
            "identify_3d_shapes_vertices_edges",
        ],
        "measurement": [
            "measure_in_inches_feet_cm",
            "time_five_minute_intervals",
            "money_word_problems",
        ],
        "data": [
            "picture_graphs_scaled",
            "bar_graphs_scaled",
            "simple_line_plots",
        ],
        "algebraic_thinking": [
            "even_odd_reasoning",
            "simple_equation_balance",
            "number_patterns_arrays",
        ],
    },

    3: {
        "description": "Grade 3 launches multiplication/division, fraction number lines, and area models.",
        "reference": "CA CCSSM Grade 3 (2023 framework)",
        "numbers": {
            "range": (0, 10000),
            "place_value": True,
            "rounding": ["nearest_10", "nearest_100"],
        },
        "operations": [
            "multiplication_facts_10x10",
            "division_facts_with_remainders",
            "two_step_word_problems",
            "distributive_relationships",
        ],
        "fractions": [
            "fraction_number_line",
            "equivalent_fractions_models",
            "compare_fractions_same_num_or_denom",
        ],
        "geometry": [
            "area_rectangles_unit_squares",
            "perimeter_word_problems",
            "classify_quadrilaterals_polygons",
        ],
        "measurement": [
            "time_to_minute_and_elapsed",
            "mass_and_volume_estimations",
            "scaled_measurement_models",
        ],
        "data": [
            "scaled_picture_graphs",
            "scaled_bar_graphs",
            "line_plots_whole_units",
        ],
        "algebraic_thinking": [
            "patterns_tables_graphs",
            "unknowns_in_equations",
            "multiplication_division_relationship",
        ],
    },

    4: {
        "description": "Grade 4 emphasizes multi-digit computation, rich fraction work, and geometry lines.",
        "reference": "CA CCSSM Grade 4 (2023 framework)",
        "numbers": {
            "range": (0, 1000000),
            "place_value": True,
            "rounding": ["nearest_10", "nearest_100", "nearest_1000"],
        },
        "operations": [
            "multi_digit_add_sub_fluency",
            "multi_digit_multiplication_standard",
            "long_division_one_digit_divisor",
            "multi_step_word_problems",
        ],
        "fractions": [
            "equivalent_fractions_number_line",
            "add_sub_fractions_like_denoms",
            "fraction_multiplication_by_whole",
            "fraction_decimal_relationships",
        ],
        "decimals": [
            "read_write_hundredths",
            "compare_order_decimals",
        ],
        "geometry": [
            "lines_rays_segments",
            "angle_measurement_protractors",
            "classify_triangles_quadrilaterals",
            "line_symmetry",
        ],
        "measurement": [
            "unit_conversions_same_system",
            "angle_measure_word_problems",
            "area_perimeter_composite_figures",
        ],
        "data": [
            "line_plots_fractional_data",
            "measurement_data_analysis",
        ],
        "algebraic_thinking": [
            "factors_and_multiples",
            "pattern_rules_and_functions",
            "equations_with_unknowns",
        ],
    },

    5: {
        "description": "Grade 5 deepens fraction/decimal operations, volume, and coordinate reasoning.",
        "reference": "CA CCSSM Grade 5 (2023 framework)",
        "numbers": {
            "range": (0, 10000000),
            "place_value": True,
            "rounding": ["nearest_10", "nearest_1000", "nearest_100000"],
        },
        "operations": [
            "multi_digit_multiplication_fluency",
            "long_division_two_digit_divisor",
            "decimal_operations_hundredths",
            "multi_step_real_world_problems",
        ],
        "fractions": [
            "add_sub_fractions_unlike_denoms",
            "add_sub_mixed_numbers",
            "multiply_fraction_by_fraction",
            "divide_unit_fractions_whole_numbers",
        ],
        "decimals": [
            "place_value_thousandths",
            "compare_round_decimals",
            "decimal_to_fraction_conversion",
        ],
        "geometry": [
            "coordinate_plane_first_quadrant",
            "classify_2d_figures_attributes",
            "volume_rectangular_prisms",
        ],
        "measurement": [
            "convert_units_same_system",
            "volume_word_problems",
            "scale_drawings_linear_measure",
        ],
        "data": [
            "line_plots_fraction_measurements",
            "interpret_data_from_conversions",
        ],
        "algebraic_thinking": [
            "write_interpret_numerical_expressions",
            "two_rule_patterns",
            "analyze_graphs_from_tables",
        ],
    },

    6: {
        "description": "Grade 6 transitions to ratios, signed numbers, expressions, and data distributions.",
        "reference": "CA CCSSM Grade 6 (2023 framework)",
        "numbers": {
            "range": (-1000000, 1000000),
            "place_value": True,
            "absolute_value": True,
        },
        "operations": [
            "integer_operations_fluency",
            "rational_number_four_operations",
            "divide_fraction_by_fraction",
            "multi_step_ratio_problems",
        ],
        "ratios": [
            "ratio_notation_and_equivalence",
            "unit_rates_tables_graphs",
            "percent_as_rate_per_100",
            "double_number_line_models",
        ],
        "algebra": [
            "write_simplify_expressions",
            "one_step_equations_inequalities",
            "dependent_independent_variables",
            "coordinate_plane_all_quadrants",
        ],
        "geometry": [
            "area_polygons_on_coordinate_plane",
            "surface_area_nets_prisms",
            "volume_with_fractional_edges",
            "right_triangle_area_relationships",
        ],
        "statistics": [
            "statistical_questions_variability",
            "dot_plots_histograms_box_plots",
            "measures_of_center_and_spread",
        ],
    },

    7: {
        "description": "Grade 7 solidifies proportional reasoning, rational operations, and probability.",
        "reference": "CA CCSSM Grade 7 (2023 framework)",
        "numbers": {
            "range": (-10000000, 10000000),
            "place_value": True,
            "absolute_value": True,
        },
        "operations": [
            "add_subtract_rational_numbers",
            "multiply_divide_rational_numbers",
            "percent_increase_decrease",
            "convert_between_forms_fraction_decimal_percent",
        ],
        "fractions": [
            "add_subtract_fractions_any_denominator",
            "multiply_divide_fractions_and_mixed_numbers",
            "complex_fraction_applications",
            "convert_fraction_decimal_percent",
        ],
        "ratios": [
            "analyze_proportional_relationships",
            "constant_of_proportionality",
            "scale_drawings_and_models",
            "multi_step_percent_problems",
        ],
        "algebra": [
            "generate_equivalent_expressions",
            "two_step_equations_and_inequalities",
            "linear_relationship_tables_graphs",
            "use_equations_to_model_situations",
        ],
        "geometry": [
            "angle_relationships_transversals",
            "area_circles_circumference",
            "surface_area_and_volume_prisms_pyramids",
            "geometric_constructions_and_cross_sections",
        ],
        "statistics": [
            "random_sampling_inferences",
            "comparative_inferences_two_populations",
            "simple_and_compound_probability",
            "uniform_probability_models",
        ],
    },

    8: {
        "description": "Grade 8 unites linear algebra, functions, transformations, and advanced geometry.",
        "reference": "CA CCSSM Grade 8 (2023 framework)",
        "numbers": {
            "range": (-100000000, 100000000),
            "place_value": True,
            "irrational_numbers": True,
            "scientific_notation": True,
        },
        "operations": [
            "integer_exponents_properties",
            "radical_simplification",
            "scientific_notation_operations",
        ],
        "algebra": [
            "solve_linear_equations_one_variable",
            "systems_of_linear_equations",
            "linear_equations_from_functions",
            "analyze_bivariate_relationships",
        ],
        "functions": [
            "define_compare_functions",
            "linear_vs_nonlinear_patterns",
            "model_with_functions",
        ],
        "geometry": [
            "translations_rotations_reflections_dilations",
            "congruence_and_similarity",
            "pythagorean_theorem_and_distance",
            "volume_cylinders_cones_spheres",
        ],
        "statistics": [
            "scatter_plots_and_trend_lines",
            "two_way_tables_relative_frequency",
            "associations_in_bivariate_data",
            "probability_models_experimental_vs_theoretical",
        ],
    }
}


def get_grade_info(grade: int):
    """
    Returns the math expectations and skill capabilities for a given grade.
    Used by topic_interpreter, problem_planner, and difficulty_scaler.
    """
    return GRADE_PROGRESSION.get(grade, None)
