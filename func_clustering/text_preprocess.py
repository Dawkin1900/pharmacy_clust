"""
Программа: Препроцессинг текста
Версия: 1.0
"""

from typing import List, Generator
import re
import gensim
import spacy


def preprocess_sentence(s: str) -> str:
    """
    Предобрабатывает текст
    
    Parameters
    ----------
    s: str
        Текст, который необходимо обработать.
        
    Returns
    -------
    str
        Возвращает предобработанный текст.
    """
    # уберем пробелы в начале и в конце строки
    s = s.strip()

    # переведем все символы в нижний регистр
    s = s.lower()

    # заменим на пробелы все символы, кроме букв
    s = re.sub(r"[^а-яА-Я]+", " ", s)

    # уберем дублирующие пробелы
    s = re.sub(r"\s{2,}", " ", s)

    # уберем пробелы в начале и в конце строки
    s = s.strip()

    return s


def sent_to_words(sentences: List[str]) -> Generator[List[str], None, None]:
    """
    Преобразует предложения в список слов.
    
    Parameters:
    ----------
    sentences : List[str]
        Список предложений.
    
    Yields:
    -------
    List[str]
        Список слов.
    """
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence),
                                              min_len=3,
                                              max_len=25,
                                              deacc=True))


def remove_stopwords(texts: List[List[str]], stop_words: list) -> List[List[str]]:
    """
    Удаляет стоп-слова из текста.
    
    Parameters:
    ----------
    texts : List[List[str]]
        Список текстов.
    stop_words: list
        Список стоп-слов.
    
    Returns:
    -------
    List[List[str]]
        Список текстов без стоп-слов.
    """
    return [[word for word in doc if word not in stop_words] for doc in texts]


def lemmatization(
    nlp: spacy.language.Language,
    stop_words: list,
    texts: List[List[str]],
    allowed_postags: List[str] = ['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN']
) -> List[List[str]]:
    """
    Лемматизирует текст.
    
    Parameters:
    ----------
    nlp: spacy.language.Language
        Объект языковой модели SpaCy.
    stop_words: list
        Список стоп-слов.
    texts : List[List[str]]
        Список текстов.
    allowed_postags : List[str], optional
        Разрешенные части речи.
    
    Returns:
    -------
    List[List[str]]
        Список лемматизированных текстов.
    """
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([
            token.lemma_ for token in doc
            if token.pos_ in allowed_postags and token.lemma_ not in stop_words
        ])
    return texts_out
