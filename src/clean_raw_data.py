
def review_to_words(raw_review):
    from konlpy.tag import Okt
    okt = Okt()
    # stop word 제거

    # 어간 추출
    try:
        stemming_words = okt.morphs(raw_review)
    except Exception:
        raise ValueError(f'Not proper value. {raw_review}')

    return ' '.join(stemming_words)

