# Project name

* 실시간 표정에 따른 배경 색 제어                


## Requirement


```
* 10th generation Intel® CoreTM processor onwards
* At least 32GB RAM
* python3.9.13
```

## Prerequite
1. 작업 폴더 생성      

```shell
mkdir work
```
2. python 가상환경 생성                 
```shell
python -m venv openvino_env
openvino_env\Scripts\activate
```
3. git clone             
```shell
git clone https://github.com/openvinotoolkit/open_model_zoo.git
git clone https://github.com/JongChanHa/Intel_AI_project01.git
```

3. pip install              
```shell
cd open_model_zoo
python -m pip install -U pip
pip install -r ./demos/common/python/requirements.txt
```

* 폴더 구조
```bash
├── open_model_zoo
├── Intel_AI_project01
└── openvino_env
``` 
## Steps to build 

1. pretrained model download&converter               
```shell
cd Intel_AI_project01
type models.lst

omz_downloader --list models.lst
omz_converter --list models.lst
```

## Steps to run

1. activate env (이미 켜져 있다면 건너뛰기)           
```shell
openvino_env/Srcipts/activate
```
2. demo실행
```shell
cd Intel_AI_project01 
python demo.py
```

## Output

![./images/result.jpg](./images/result.jpg)

## Appendix

* (참고 자료 및 알아두어야할 사항들 기술)
