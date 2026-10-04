from src.preprocessing import preprocess_data, print_result, save_preprocessed_data
from src.feature_selection import (
    load_preprocessed,
    split_data,
    save_split,
    print_split_result,
    scale_data,
    save_scaled,
    print_scaling_result,
    select_features_mi,
    save_mi_selected,
    print_mi_result,
    select_features_l1,
    save_l1_selected,
    print_l1_result,
)


def main():
    # --- 1. Preprocessing (Ahmad Mathlaul Falah) ---
    X, y, info = preprocess_data()
    output_path = save_preprocessed_data(X, y)
    print_result(X, y, info)
    print(f"\nData hasil preprocessing disimpan di: {output_path}")

    # --- 2. Split data (Ah. Dliya'ul Adlha Jamalul Lail) ---
    print()
    X, y = load_preprocessed(output_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    split_paths = save_split(X_train, X_test, y_train, y_test)
    print_split_result(X_train, X_test, y_train, y_test)
    print("\nFile split disimpan di:")
    for k, v in split_paths.items():
        print(f"  {k}: {v}")

    # --- 3. Min-Max scaling (Ah. Dliya'ul Adlha Jamalul Lail) ---
    print()
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    scaled_paths = save_scaled(X_train_scaled, X_test_scaled)
    print_scaling_result(X_train_scaled, X_test_scaled, scaler)
    print("\nFile scaled disimpan di:")
    for k, v in scaled_paths.items():
        print(f"  {k}: {v}")

    # --- 4. Mutual Information (Ah. Dliya'ul Adlha Jamalul Lail) ---
    print()
    X_train_mi, X_test_mi, mi_series = select_features_mi(
        X_train_scaled, X_test_scaled, y_train
    )
    mi_paths = save_mi_selected(X_train_mi, X_test_mi)
    print_mi_result(X_train_mi, X_test_mi, mi_series)
    print("\nFile MI selected disimpan di:")
    for k, v in mi_paths.items():
        print(f"  {k}: {v}")

    # --- 5. L1-based Feature Selection (Ah. Dliya'ul Adlha Jamalul Lail) ---
    print()
    X_train_l1, X_test_l1, coef_series, selector = select_features_l1(
        X_train_scaled, X_test_scaled, y_train
    )
    l1_paths = save_l1_selected(X_train_l1, X_test_l1)
    print_l1_result(X_train_l1, X_test_l1, coef_series)
    print("\nFile L1 selected disimpan di:")
    for k, v in l1_paths.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
