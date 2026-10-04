from src.preprocessing import preprocess_data, print_result, save_preprocessed_data


def main():
    X, y, info = preprocess_data()
    output_path = save_preprocessed_data(X, y)

    print_result(X, y, info)
    print(f"\nData hasil preprocessing disimpan di: {output_path}")


if __name__ == "__main__":
    main()
