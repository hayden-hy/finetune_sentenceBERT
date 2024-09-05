def pdfToCleanText(pdfPath : str = "Data/EU.pdf", outputPath : str = "Data/corpus.txt", pages : list = None, returnCorpus : bool = True) -> str:
    """Convert pdf to clean text

    Args:
        pdfPath (str, optional): Path to pdf file. Defaults to "rawData/EU.pdf".
        outputPath (str, optional): Path to output file. Defaults to "data/EU.txt".
        pages (list, optional): List of pages to convert. Defaults to None.
        returnCorpus (bool, optional): Preprocess text. Defaults to True. It will return a set of sentences.

    Returns:
        str: Clean text from pdf
    """
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfpage import PDFPage
    from io import StringIO
    import re

    if pages == None:
        pagenums = set(list(range(2, 59)) + list(range(61, 104)))
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = open(pdfPath, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close()

    REGEX_ROMAN_NUMBERS = "[ ]([mdclxvi]+ )"
    REGEX_MULTIPLE_SPACES = "\s\s+"

    if returnCorpus:
        text = text.lower()

        text = re.sub("[(\{\[][a-z]{0,3}[)\}\]]|[0-9][a-z]?[.]|(p. [0-9])", "", text)
        text = re.sub(REGEX_ROMAN_NUMBERS, "", text)

        # remove \n, \t, \r
        text = text.replace("\n", "").replace("\t", "").replace("\r", "").replace("\x0c", "").replace("\x0b", "")
        # remove (, ), [, ], {, }, <, >, ", ', :, ;, ., ,, ?, !, -, _, /, \, |, &, @, #, $, %, ^, *, +, =, ~, `
        text = text.replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("<", "").replace(">", "").replace('"', "").replace("'", "").replace(":", "").replace(";", "").replace(",", "").replace("_", "").replace("/", "").replace("\\", "").replace("|", "").replace("&", "").replace("@", "").replace("#", "").replace("$", "").replace("%", "").replace("^", "").replace("*", "").replace("+", "").replace("=", "").replace("~", "").replace("`", "").replace('15698/22', '').replace('”', '').replace('‘', '').replace('’', '').replace(' ’ ', "").replace('RB/ek','').replace('TREE.2.B', '')
        # remove numbers 
        text = "".join([i for i in text if not i.isdigit()])
        # remove majuscules

        text = text.replace('tree.b', '').replace('rbek', '').replace('treeb', '').replace(' ec ', '')

        # remove multiple spaces
        text = " ".join(text.split())

        test = re.sub(REGEX_MULTIPLE_SPACES, " ", text)

        from nltk.tokenize import sent_tokenize
        from nltk import download
        try:
            text = sent_tokenize(text)
        except Exception as e:
            download("punkt")
            text = sent_tokenize(text)
        print("Nombre de phrases : " + str(len(text)))
        # # remove small sentences
        # text = [sentence for sentence in text if len(sentence) > 10]
        # split sentences with spaces
        text = [sentence.split() for sentence in text]
        # # add group of 2 words
        # maxLengthWord = 4
        # for i in range(len(text)):
        #     for k in range(2, maxLengthWord + 1):
        #         temp = []
        #         for j in range(0, len(text[i]) - k + 1):
        #             temp.append(" ".join(text[i][j:j+k]))
        #         text.append(temp)
  
    else:
        text = text.split()

    with open(outputPath, "w") as f:
        for sentence in text:
            f.write(" ".join(sentence) + "\n")
    return text


