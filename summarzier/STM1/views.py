from django.shortcuts import render
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
from matplotlib.pyplot import text
import time
import math
import speech_recognition as sr
from textProcessingModule import *
from Ngrams import *
from TfIdfModule import *
from CosineModule import CosineSimilarity
from operator import itemgetter
from transformers import T5ForConditionalGeneration, T5Tokenizer # install these and also torch and sentencepiece in environment
# Create your views here.
def home(request):
    return render (request,'home.html')
def transcript(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES ['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
    return render (request=request, template_name='transcript.html')
    #return render (request=request, template_name="register.html", context={"register_form":form})

def summarize(request):
    begin = time.time()
    if request.method == 'POST':
        #file = request.POST.get('file')
        file = request.FILES['file']
    if file != '':
        transcript1 = file.read()
        inputs = str(transcript1)
    # initialize the model architecture and weights
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    # initialize the model tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    inputs2 = '"{}"'.format(inputs)
    inputs1 = tokenizer.encode("summarize: " + inputs2, return_tensors="pt", max_length=2000, truncation=True)
    # generate the summarization output
    outputs = model.generate(
        inputs1,
        max_length=1500,
        min_length=70,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True)

# just for debugging
    #return outputs
    print(tokenizer.decode(outputs[0]))
    output1 = tokenizer.decode(outputs[0])
    #length of input
    input_len = len(inputs2)
    #length of summary
    summ_len =  len(output1)
    #ratio of input:output
    div=math.gcd(input_len,summ_len)
    ratio = f"Ratio(input/output) = {input_len//div}:{summ_len//div}"
    #Percentage compression
    sum_compress = 100 - ((summ_len/input_len)*100)
    #print(summ_compress, "%")
    #time taken for summarizing
    time.sleep(1)
    end = time.time()
    ptime = f"Processing time = {end - begin}"
    return render (request, 'transcript.html', {"text":output1,"compress": sum_compress,"input":input_len,"summ": summ_len , "ratio": ratio,"time": ptime })

    #"input":input_len, "summ": summ_len , "ratio": ratio, "time": "time": ptime

def extract_page(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES ['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
    return render (request=request, template_name='extract_transcript.html')

def extractive_sum(request):
    begin = time.time()

    if request.method == 'POST':
        #file = request.POST.get('file')
        file = request.FILES['file']
    if file != '':
        transcript1 = file.read()
        text = str(transcript1)
    nGramNumber = 3
    '''
    Pre-processing of the Text
    '''
    textProcessing = TextProcessing()
    tokenizedSentence = textProcessing.sentenceTokenization(text)

    noOfSentences = 0.50*len(tokenizedSentence)
    #print(tokenizedSentence)

    text = textProcessing.preprocessText(text)
    tokenizedWords = textProcessing.tokenizing(text)


    nGrams = Ngrams()
    #print("tokenizedWords ->",tokenizedWords)
    tokenizedWordsWithNGrams = nGrams.generate_ngrams(tokenizedWords, nGramNumber)
    #print(tokenizedWordsWithNGrams)

    tfId = TfIdf(nGramNumber)
    tfIdMatrix = tfId.calculateTfIdfMatrix(tokenizedWordsWithNGrams, tokenizedSentence)

    #print(tfIdMatrix)
    outF = open('output.txt',"w")
    outF.write(tfIdMatrix.__str__())

    cosine = CosineSimilarity()
    sentenceImportanceValues = cosine.calculateCosineSimilarity(tfIdMatrix)

    sentenceImportanceValues = sorted(sentenceImportanceValues.items(), key = itemgetter(1), reverse=True)
    ##print(sentenceImportanceValues)

    cnt = 0
    sentenceNo = []

    #print("no. of sentences ->",noOfSentences)

    for sentence_prob in sentenceImportanceValues:
        if cnt <= noOfSentences:
            sentenceNo.append(sentence_prob[0])
            cnt = cnt + 1
        else:
            break

    sentenceNo.sort()
    #print(sentenceNo)

    summary = []
    #print("Sentence->",tokenizedSentence)

    for value in sentenceNo:
        summary.append(tokenizedSentence[value])

    summary = " ".join(summary)
    #length of input
    input_len = len(text)
    #length of summary
    summ_len =  len(summary)
    #ratio of input:output
    div=math.gcd(input_len,summ_len)
    ratio = f"Ratio(input/output) = {input_len//div}:{summ_len//div}"
    #Percentage compression
    summ_compress = 100 - ((summ_len/input_len)*100)
    #print(summ_compress, "%")
    #time taken for summarizing
    time.sleep(1)
    end = time.time()
    ptime = f"Processing time = {end - begin}"

    return render (request, 'extract_transcript.html',{"text":summary,"compress": summ_compress,"input":input_len,"summ": summ_len , "ratio": ratio,"time": ptime })
def audio_page(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES ['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
    return render (request=request, template_name='audio.html')
def audio_sum(request):
    begin = time.time()
    if request.method == 'POST':
        file = request.FILES['file']
   # Initialize recognizer class (for recognizing the speech)
    r = sr.Recognizer()

# Reading Audio file as source
# listening the audio file and store in audio_text variable

    with sr.AudioFile(file) as source:
        audio_text = r.listen(source)

# recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    try:

        # using google speech recognition
        text= r.recognize_google(audio_text)
        print('Converting audio transcripts into text ...')
        print(text)

    except:
         print('Sorry.. run again...')
    nGramNumber = 3
    '''
    Pre-processing of the Text
    '''
    textProcessing = TextProcessing()
    tokenizedSentence = textProcessing.sentenceTokenization(text)

    noOfSentences = 0.50*len(tokenizedSentence)
    #print(tokenizedSentence)

    text = textProcessing.preprocessText(text)
    tokenizedWords = textProcessing.tokenizing(text)


    nGrams = Ngrams()
    tokenizedWordsWithNGrams = nGrams.generate_ngrams(tokenizedWords, nGramNumber)
    #print(tokenizedWordsWithNGrams)

    tfId = TfIdf(nGramNumber)
    tfIdMatrix = tfId.calculateTfIdfMatrix(tokenizedWordsWithNGrams, tokenizedSentence)

    #print(tfIdMatrix)
    outF = open('output.txt',"w")
    outF.write(tfIdMatrix.__str__())

    cosine = CosineSimilarity()
    sentenceImportanceValues = cosine.calculateCosineSimilarity(tfIdMatrix)

    sentenceImportanceValues = sorted(sentenceImportanceValues.items(), key = itemgetter(1), reverse=True)
    ##print(sentenceImportanceValues)

    cnt = 0
    sentenceNo = []

    #print("no. of sentences ->",noOfSentences)

    for sentence_prob in sentenceImportanceValues:
        if cnt <= noOfSentences:
            sentenceNo.append(sentence_prob[0])
            cnt = cnt + 1
        else:
            break

    sentenceNo.sort()
    #print(sentenceNo)

    summary = []
    #print("Sentence->",tokenizedSentence)

    for value in sentenceNo:
        summary.append(tokenizedSentence[value])

    summary = '" "'.join(summary)

    return render (request, 'audio.html',{"text":summary})






        


