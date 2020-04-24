ROOT=$1
USER=$2
PASSWD=$3

echo Root: $ROOT
echo User: $USER
echo Password: $PASSWD

ROOT=${ROOT}/iam
USER=--user=${USER}
PASSWD=--password=${PASSWD}

PWD=$(pwd)

mkdir --parents "${ROOT}"
cd "${ROOT}" || exit
wget ${USER} ${PASSWD} -nc www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip
unzip -o largeWriterIndependentTextLineRecognitionTask.zip -d largeWriterIndependentTextLineRecognitionTask
cd "$PWD" || exit

mkdir --parents "${ROOT}/data/ascii"
cd "${ROOT}/data/ascii" || exit
wget ${USER} ${PASSWD} -nc http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/ascii.tgz
7z x ascii.tgz -so | 7z x -aoa -si -ttar -o"./"
cd "${PWD}" || exit

mkdir --parents "${ROOT}/data/words"
cd "${ROOT}/data/words" || exit
wget ${USER} ${PASSWD} -nc http://www.fki.inf.unibe.ch/DBs/iamDB/data/words/words.tgz
7z x words.tgz -so | 7z x -aoa -si -ttar -o"./"
cd "$PWD" || exit


