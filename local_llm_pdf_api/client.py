import requests

BASE_URL = "http://localhost:8000"

PDF_PATH = "sample.pdf"

QUESTION = "What is the main topic of this PDF?"


def load_model():
    print("\nCalling /load-model API...")

    response = requests.post(BASE_URL + "/load-model")

    print("Status Code:", response.status_code)
    print("Response:", response.json())


def ask_pdf():
    print("\nCalling /ask-pdf API...")

    with open(PDF_PATH, "rb") as pdf_file:
        files = {
            "file": pdf_file
        }

        data = {
            "question": QUESTION
        }

        response = requests.post(
            BASE_URL + "/ask-pdf",
            files=files,
            data=data
        )

    print("Status Code:", response.status_code)

    result = response.json()

    print("\nQUESTION:")
    print(result.get("question"))

    print("\nANSWER:")
    print(result.get("answer"))

    print("\nFurther Processing Example:")
    answer = result.get("answer", "")
    print("Answer word count:", len(answer.split()))


if __name__ == "__main__":
    load_model()
    ask_pdf()
