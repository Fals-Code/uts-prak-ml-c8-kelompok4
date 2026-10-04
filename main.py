from src.preprocessing import preprocess_data, print_result, save_preprocessed_data
from src.feature_selection import (
    load_preprocessed,
    split_data,
    save_split,
    print_split_result,
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


if __name__ == "__main__":
    main()
