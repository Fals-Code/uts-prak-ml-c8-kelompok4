from src.preprocessing import preprocess_data, print_result


def main():
    X, y, info = preprocess_data()
    print_result(X, y, info)


if __name__ == "__main__":
    main()
