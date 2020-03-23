import argparse
import joblib
import pandas as pd

MODEL_PATH = "iris.joblib"
LABEL_ENCODER_PATH = "le.joblib"


def main():
    parser = argparse.ArgumentParser(
        description='Prediction iris dataset',
    )
    parser.add_argument(
        '--input-file',
        required=True,
        help='Input file',
        type=str,
        dest='input_file',
    )
    parser.add_argument(
        '--output-file',
        required=True,
        help='Output file',
        type=str,
        dest='output_file',
    )
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        df = pd.read_csv(f)

    df["sepal-square"] = df["sepal-length"] * df["sepal-width"]
    df["petal-square"] = df["petal-length"] * df["petal-width"]

    with open(MODEL_PATH, "rb") as file:
        model = joblib.load(file)

    with open(LABEL_ENCODER_PATH, "rb") as file:
        le = joblib.load(file)

    predicted_label = model.predict(df.values)
    predicted = le.inverse_transform(predicted_label)

    with open(args.output_file, 'w') as f:
        for item in predicted:
            f.write(item + '\n')


if __name__ == "__main__":
    main()
