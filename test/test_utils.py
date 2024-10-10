import sys
sys.path.append('./src')
from utils import countDown, getStartIndex,putIndexInFormatMessage


def test_getStartIndex():
    assert getStartIndex("sec until taking photo...") == ([],0)
    assert getStartIndex("{} sec until taking photo...") == ([0],1)
    assert getStartIndex("{0} sec until taking photo...") == ([],1)
    assert getStartIndex("{0} sec {} until taking photo...") == ([1],2)
    assert getStartIndex("{0} sec {} until taking photo...{}") == ([1,2],3)

def test_putIndexInFormatMessage():
    assert putIndexInFormatMessage([0],"{} sec until taking photo...") == "{0} sec until taking photo..."
    assert putIndexInFormatMessage([1],"{0} sec until {} taking photo...") == "{0} sec until {1} taking photo..."

def test_stringFormat(capsys):
    countDown(5,"{0} sec {}","until taking photo...")
    captured = capsys.readouterr()

    testMessage = [
        "5 sec until taking photo...",
        "4 sec until taking photo...",
        "3 sec until taking photo...",
        "2 sec until taking photo...",
        "1 sec until taking photo..."
    ]
    targetMessage = captured.out.strip("\n").split("\n")
    with capsys.disabled():
        print(targetMessage)
    assert targetMessage == testMessage

