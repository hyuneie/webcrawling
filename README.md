
# 웹 크롤링을 통한 GS,CU 편의점 간 감성분석을 진행해보았습니다.

1. 트위터는 API를 사용할 경우 일주일 전까지의 데이터밖에 분석이 안되기에 외부 모듈을 사용하여 일년치 데이터를 합쳤습니다
> 네이버는 API를 사용하여 GS,CU 편의점의 데이터를 모았습니다.

2. 크롤링 시 편의점을 지칭하는 단어가 다르기에 GS편의점, 지에스편의점, 지에스, gs, cu, 씨유 등 한글과 영어를 혼용하여 크롤링을 진행했습니다.

> 크롤링 진행 후 불용어, 의미없는 단어 등을 제거하는 데이터 전처리를 진행하였습니다. 
> 중복되어 나오는 광고문구나 바이럴 문구등을 제거할까 고민했지만 광고나 바이럴이 진행되는 것은 역으로 이 편의점에 대한 관심도가 증가하는 것으로 판단하여 제거하지 않았습니다.

3. 이후 크롤링 및 전처리가 완료 된 데이터를 대상으로 감성분석을 진행하였습니다. 
