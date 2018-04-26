import corenlp
import numpy as np
from nltk.tree import Tree
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import json
import queue
import random

#constant definition
STOPWORDS = stopwords.words('english')
PUNCTUATIONS = [',','.',':',';','?','(',')','[',']','&','!','*','@','#','$','%']
QUESTIONTYPE = ['what','who','where','when','how','why','which','other']
TIME = ['DATE','DURATION','TIME']
NUMERICAL = ['MONEY','NUMBER','ORDINAL','PERCENT'] 
MAX_SIZE = 100

#structure definition
class question_span:
    def __init__(self):
        self.content = ""
        self.substiSection = ""
        self.questiontype = ""

class answer_span:
    def __init__(self):
        self.tokens = []
        self.answertype = ""

class text_span:
    def __init__(self):
        self.tokens = []

#Get tokens corresponding to an NP non-terminals from a document parse Tree 
#e.g. given [NP([NP([DT([The])],[NNS([writings])])],[PP([IN([of])],[NP([NNP(Samuel)],[NNP(Pepys)])])])]
#e.g. given [NP([NNP(Samuel)],[NNP(Pepys)])]
#the function will return [Samuel Pepys] as an integration
def getNPFromTree(root):
    result = ""
    if len(root.child) > 0:
        for node in root.child:
            if node.value == "NP":
                return "", False
            res, found = getNPFromTree(node)
            if not found:
                return "", False
            else:
                result += res + " "
        return result, True
    else:
        return root.value, True

#Get tokens corresponding to an WHNP non-terminals from a document parse Tree 
#e.g. given child [WHNP([WDT(Which)],[NN(prize)])]
#the fucntion will return [Which prize] as an integration
def getWHNPFromTree(root):
    result = ""
    if len(root.child) > 0:
        for node in root.child:
            if node.value == "WHNP":
                return "", False
            res, found = getNPFromTree(node)
            if not found:
                return "", False
            else:
                result += res + " "
        return result, True
    else:
        return root.value, True

#Process the parse tree to get all NP tokens
#e.g. given a sentence that "The writings of Samuel Pepys describe the pub as the heart of England."
#the function will return [[The writings], [the pub], [Samuel Pepys], [the heart], [England]] consisting of 5 NP 
def getAllNPTokens(bfs):
    np = queue.Queue(maxsize=MAX_SIZE)
    while not bfs.empty():
        node = bfs.get()
        if node.value == "NP":
            np.put(node)
        if len(node.child) > 0:
            for child in node.child:
                bfs.put(child)
    
    nps = []
    while not np.empty():
        node = np.get()
        res, found = getNPFromTree(node)
        if found:
            nps.append(res.split())
    
    return nps

#Process the parse tree to get all WHNP tokens
#e.g. given a sentence that "Which prize did Frederick Buechner create?"
#the function will return [Which prize] which is a single WHNP
def getAllWHNPTokens(bfs):
    whnp = queue.Queue(maxsize=MAX_SIZE)
    while not bfs.empty():
        node = bfs.get()
        if node.value == "WHNP":
            whnp.put(node)
        if len(node.child) > 0:
            for child in node.child:
                bfs.put(child)
    
    whnps = []
    while not whnp.empty():
        node = whnp.get()
        res, found = getWHNPFromTree(node)
        if found:
            whnps.append(res.split())
    
    return whnps

#Generate a candidate answer consists of answer type and answer tokens
#e.g. ["Samuel", "Pepys"] PERSON
#e.g. ["the", "pub"] O
def genCandidateAnswerSpans(l1, l2):
    count = defaultdict(int)
    spans = []
    
    for sentence in l1:
        span = answer_span()
        for token in sentence:
            span.tokens.append(token)
            for token2 in l2:
                t = token2.word
                if token == t:
                    if token2.ner != "0":
                        span.answertype = token2.ner
                    break
        spans.append(span)
    return spans

def genSpans(l1, l2):
    count = defaultdict(int)
    spans = []
    
    for sentence in l1:
        span = text_span()
        for token in sentence:
            span.tokens.append(token)
            for token2 in l2:
                t = token2.word
                if token == t:
                    break
        spans.append(span)
    return spans

#Since when we attempt to select a best answer from candidates 
#we will replace tokens in WH- or How with each candidates to get similarity score using tf-idf,
#we need to find a valid part to replace.
#e.g. Question: "Which prize did Frederick Buechner create?"
#The function will return "Which prize".
#Then we will replace "which prize " with each candidates, then we get a new sentence "Buechner Prize for Preaching did Frederick Buechner create"
def getReplacedPart (question, questionType):
    target = ""
    q = queue.Queue(maxsize=MAX_SIZE)
    tree = question.parseTree
    q.put(tree)
    
    tokens = getAllWHNPTokens(q)
    spans = genSpans(tokens, question.token)
    
    if questionType == 6:
        target = "which"
        index = 0
        for j in range(len(spans)):
            tempSentence = " ".join(spans[j].tokens)
            if "which" in tempSentence:
                index = j
                break
        if spans:
            target = " ".join(spans[index].tokens)
        else:
            target = "which"
    elif questionType == 7:
        target = "what"
        index = 0
        for j in range(len(spans)):
            tempSentence = " ".join(spans[j].tokens)
            if "what" in tempSentence:
                index = j
                break
        if spans:
            target = " ".join(spans[index].tokens)
        else:
            target = "what"
    
    return target

def getBestAnswer(caSpans, qSpans, context, model):
    score = []
    indices = []
    resi = 0
    
    questionType = qSpans.questionType
    question = qSpans.content
    
    if questionType == 1:
        for j in range(len(caSpans)):
            caSpan = caSpans[j]
            tokens = caSpan.tokens
            catype = caSpan.answertype
            if catype == "PERSON":
                tempStr = question.replace('who', " ".join(tokens))
                x = model.fit_transform([context, tempStr])
                matrix = (x * x.T).A
                score.append(matrix[0][1])
                indices.append(j)
        score = np.array(score)
        if len(score) > 0:
            maxi = np.argmax(score)
            resi = indices[maxi]
        else:
            try:
                resi = random.randint(0, len(caSpans) - 1)
            except Exception as e:
                pass
    
    elif questionType == 2:
        for j in range(len(caSpans)):
            caSpan = caSpans[j]
            tokens = caSpan.tokens
            catype = caSpan.answertype
            if catype == "LOCATION" or catype == "ORGANIZATION":
                tempStr = question.replace('where', " ".join(tokens))
                x = model.fit_transform([context, tempStr])
                matrix = (x * x.T).A
                score.append(matrix[0][1])
                indices.append(j)
        score = np.array(score)
        if len(score) > 0:
            maxi = np.argmax(score)
            resi = indices[maxi]
        else:
            try:
                resi = random.randint(0, len(caSpans) - 1)
            except Exception as e:
                pass
            
    elif questionType == 3:
        for j in range(len(caSpans)):
            caSpan = caSpans[j]
            tokens = caSpan.tokens
            catype = caSpan.answertype
            if catype in TIME:
                tempStr = question.replace('when', " ".join(tokens))
                x = model.fit_transform([context, tempStr])
                matrix = (x * x.T).A
                score.append(matrix[0][1])
                indices.append(j)
        score = np.array(score)
        if len(score) > 0:
            maxi = np.argmax(score)
            resi = indices[maxi]
        else:
            try:
                resi = random.randint(0, len(caSpans) - 1)
            except Exception as e:
                pass
    
    elif questionType == 6:
        for j in range(len(caSpans)):
            caSpan = caSpans[j]
            tokens = caSpan.tokens
            catype = caSpan.answertype
            tempStr = question.replace(qSpans.substiSection, " ".join(tokens))
            x = model.fit_transform([context, tempStr])
            matrix = (x * x.T).A
            score.append(matrix[0][1])
            indices.append(j)
        score = np.array(score)
        if len(score) > 0:
            maxi = np.argmax(score)
            resi = indices[maxi]
        else:
            try:
                resi = random.randint(0, len(caSpans) - 1)
            except Exception as e:
                pass
    
    elif questionType == 7:
        for j in range(len(caSpans)):
            caSpan = caSpans[j]
            tokens = caSpan.tokens
            catype = caSpan.answertype
            tempStr = question.replace(qSpans.substiSection, " ".join(tokens))
            x = model.fit_transform([context, tempStr])
            matrix = (x * x.T).A
            score.append(matrix[0][1])
            indices.append(j)
        score = np.array(score)
        if len(score) > 0:
            maxi = np.argmax(score)
            resi = indices[maxi]
        else:
            try:
                resi = random.randint(0, len(caSpans) - 1)
            except Exception as e:
                pass
    
    elif questionType == 4:
        if "how many" in question:
            for j in range(len(caSpans)):
                caSpan = caSpans[j]
                tokens = caSpan.tokens
                catype = caSpan.answertype
                if catype in NUMERICAL:
                    tempStr = question.replace("how many", " ".join(tokens))
                    x = model.fit_transform([context, strReplaced])
                    matrix = (x * x.T).A
                    score.append(matrix[0][1])
                    indices.append(j)
                    
        elif "how much" in question:
            for j in range(len(caSpans)):
                caSpan = caSpans[j]
                tokens = caSpan.tokens
                catype = caSpan.answertype
                if catype in NUMERICAL:
                    tempStr = question.replace("how much", " ".join(tokens))
                    x = model.fit_transform([context, strReplaced])
                    matrix = (x * x.T).A
                    score.append(matrix[0][1])
                    indices.append(j)
        
        elif "how long" in question:
            for j in range(len(caSpans)):
                caSpan = caSpans[j]
                tokens = caSpan.tokens
                catype = caSpan.answertype
                if catype in NUMERICAL or catype in TIME:
                    tempStr = question.replace("how long", " ".join(tokens))
                    x = model.fit_transform([context, strReplaced])
                    matrix = (x * x.T).A
                    score.append(matrix[0][1])
                    indices.append(j)
        
        elif "how old" in question:
            for j in range(len(caSpans)):
                caSpan = caSpans[j]
                tokens = caSpan.tokens
                catype = caSpan.answertype
                if catype in NUMERICAL or catype in TIME:
                    tempStr = question.replace("how old", " ".join(tokens))
                    x = model.fit_transform([context, strReplaced])
                    matrix = (x * x.T).A
                    score.append(matrix[0][1])
                    indices.append(j)
        
        elif "how far" in question:
            for j in range(len(caSpans)):
                caSpan = caSpans[j]
                tokens = caSpan.tokens
                catype = caSpan.answertype
                if catype in NUMERICAL:
                    tempStr = question.replace("how far", " ".join(tokens))
                    x = model.fit_transform([context, strReplaced])
                    matrix = (x * x.T).A
                    score.append(matrix[0][1])
                    indices.append(j)
        score = np.array(score)
        if len(score) > 0:
            maxi = np.argmax(score)
            resi = indices[maxi]
        else:
            try:
                resi = random.randint(0, len(caSpans) - 1)
            except Exception as e:
                pass
        
    elif questionType == 0:
        for j in range(len(caSpans)):
            caSpan = caSpans[j]
            tokens = caSpan.tokens
            catype = caSpan.answertype
            tempStr = question.replace('what', " ".join(tokens))
            x = model.fit_transform([context, tempStr])
            matrix = (x * x.T).A
            score.append(matrix[0][1])
            indices.append(j)
        score = np.array(score)
        if len(score) > 0:
            maxi = np.argmax(score)
            resi = indices[maxi]
        else:
            try:
                resi = random.randint(0, len(caSpans) - 1)
            except Exception as e:
                pass
    
    else:
        try:
            resi = random.randint(0, len(caSpans) - 1)
        except Exception as e:
                pass
    
    return resi

#input and output
answerSet = defaultdict(str)
inputFile = 'testInput.json'
outputFile = 'result.json'

#read input file
with open(inputFile, 'r') as file:
    text = file.read()
data = json.loads(text)

#main function
with corenlp.CoreNLPClient(annotators='tokenize ssplit parse lemma pos ner'.split()) as client:
    for document in data['data']:
        paragraphs = defaultdict(list)
        rawParagraphs = document['paragraphs']
        for rawParagraph in rawParagraphs:
            rawContext = rawParagraph['context']
            qas = rawParagraph['qas']
            
            #process context
            #filter out punctuations and transfer it to lowercase
            tempContext = client.annotate(rawContext)
            context = []
            for s in tempContext.sentence:
                sentence = []
                for token in s.token:
                    tempToken = token.lemma.lower()
                    if tempToken not in PUNCTUATIONS:
                        sentence.append(tempToken)
                context.append(" ".join(sentence))
            
            unigramModel = TFIDF(input=context, analyzer='word', dtype=np.float32, stop_words=STOPWORDS)
            
            for qa in qas:
                rawQuestion = qa['question']
                qid = qa['id']
                
                tempQuestion = client.annotate(rawQuestion)
                question = []
                
                tokens = tempQuestion.sentence[0].token
                isIdentified = False
                questionType = 8
                questionSpan = question_span()
                for token in tokens:
                    tempToken = token.lemma.lower()
                    if tempToken not in PUNCTUATIONS:
                        question.append(tempToken)
                        if not isIdentified:
                            if token == 'what':
                                isIdentified = True
                                pos = token.pos
                                if pos == 'WP':
                                    questionType = 0
                                elif pos == 'WDT':
                                    questionType = 7
                                    questionSpan.substiSection = getReplacedPart(tempQuestion.sentence[0], questionType)
                                else:
                                    isIdentified = False
                            elif token == 'who':
                                isIdentified = True
                                questionType = 1
                            elif token == 'where':
                                isIdentified = True
                                questionType = 2
                            elif token == 'when':
                                isIdentified = True
                                questionType = 3
                            elif token == 'how':
                                isIdentified = True
                                questionType = 4
                            elif token == 'why':
                                isIdentified = True
                                questionType = 5
                            elif token == 'which':
                                isIdentified = True
                                questionType = 6
                                questionSpan.substiSection = getReplacedPart(tempQuestion.sentence[0], questionType)
                questionSpan.content = " ".join(question)
                questionSpan.questionType = questionType
                
                findMaxSimilarity = []
                for con in context:
                    ques = questionSpan.content
                    combo = [con, ques]
                    matrix = unigramModel.fit_transform(combo)
                    tempScore = (matrix * matrix.T).A
                    findMaxSimilarity.append(tempScore[0][1])
                findMaxSimilarity = np.array(findMaxSimilarity)
                maxi = np.argmax(findMaxSimilarity)
                
                candidateSentence = tempContext.sentence[maxi]
                parseTree = candidateSentence.parseTree
                
                q = queue.Queue(maxsize=MAX_SIZE)
                q.put(parseTree)
                candidateAnswers = getAllNPTokens(q)
                
                candidateAnswerSpans = genCandidateAnswerSpans(candidateAnswers, candidateSentence.token)
                size = len(candidateAnswerSpans)
                
                answerStr = ""
                if len(candidateAnswerSpans) > 0:
                    resi = getBestAnswer(candidateAnswerSpans, questionSpan, context[maxi], unigramModel)
                    answerSpan = candidateAnswerSpans[resi]
                    
                    for j in range(len(answerSpan.tokens) - 1):
                        answerStr += answerSpan.tokens[j] + " "
                    answerStr += answerSpan.tokens[len(answerSpan.tokens) - 1]
                answerSet[qid] = answerStr

resultJSON = json.dumps(answerSet)
with open(outputFile, 'w') as fout:
    fout.writelines(resultJSON)