

# Read This ON our own risk 
def evaluate_kan(InputText):
    from deep_translator import GoogleTranslator

    outputText = GoogleTranslator(source="en", target="kn").translate(InputText)
    return outputText

def evaluate_eng(InputText):
    from deep_translator import GoogleTranslator
    outputText = GoogleTranslator(source="kn", target="en").translate(InputText)
    return outputText


def main():
    pass

if __name__ == "__main__":
    main()
