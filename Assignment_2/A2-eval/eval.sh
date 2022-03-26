Submission=$(pwd)/Submission #contain tar.gz file
INPUT_DIR=$(pwd)/Input #Stores the testcases.
SRC_DIR=$(pwd)/uncomp #untar the source code.

RED='\033[0;31m'
NC='\033[0m'
GREEN='\033[0;32m'

if [ $# -lt 1 ]
then
echo "execute sh eval.sh Roll_No"
exit
fi
# Extract student directory
ROLL_NO=$1
mkdir ${SRC_DIR}/${ROLL_NO}
tar -xzf ${Submission}/${ROLL_NO}.tar.gz -C ${SRC_DIR}/${ROLL_NO}
cp "makefile" ${SRC_DIR}/${ROLL_NO}
cd ${SRC_DIR}/${ROLL_NO}/
rm -f a.out
make 
if [ $? -ne 0 ]
then
echo "Make failed!"
exit
fi

Marks=0
echo "Evaluating $ROLL_NO"
echo "*******************"
count=0
./output ${INPUT_DIR}/input1.txt Out1.txt
val=$(diff Out1.txt ${INPUT_DIR}/output1.txt | wc -l)
if [ $val -eq 0 ]
then
count=$(echo "${count} + 1" | bc -l)
echo -e "testcase $no: ${GREEN}passed${NC}"
else
echo -e "testcase $no: ${RED}failed${NC}"
fi
./output ${INPUT_DIR}/input2.txt Out2.txt
val=$(diff Out2.txt ${INPUT_DIR}/output2.txt | wc -l)
if [ $val -eq 0 ]
then
count=$(echo "${count} + 1" | bc -l)
echo -e "testcase $no: ${GREEN}passed${NC}"
else
echo -e "testcase $no: ${RED}failed${NC}"
fi
./output ${INPUT_DIR}/input3.txt Out3.txt
val=$(diff Out3.txt ${INPUT_DIR}/output3.txt | wc -l)
if [ $val -eq 0 ]
then
count=$(echo "${count} + 1" | bc -l)
echo -e "testcase $no: ${GREEN}passed${NC}"
else
echo -e "testcase $no: ${RED}failed${NC}"
fi
./output ${INPUT_DIR}/input4.txt Out4.txt
val=$(diff Out4.txt ${INPUT_DIR}/output4.txt | wc -l)
if [ $val -eq 0 ]
then
count=$(echo "${count} + 1" | bc -l)
echo -e "testcase $no: ${GREEN}passed${NC}"
else
echo -e "testcase $no: ${RED}failed${NC}"
fi
./output ${INPUT_DIR}/input5.txt Out5.txt
val=$(diff Out5.txt ${INPUT_DIR}/output5.txt | wc -l)
if [ $val -eq 0 ]
then
count=$(echo "${count} + 1" | bc -l)
echo -e "testcase $no: ${GREEN}passed${NC}"
else
echo -e "testcase $no: ${RED}failed${NC}"
fi
./output ${INPUT_DIR}/input6.txt Out6.txt
val=$(diff Out6.txt ${INPUT_DIR}/output6.txt | wc -l)
if [ $val -eq 0 ]
then
count=$(echo "${count} + 1" | bc -l)
echo -e "testcase $no: ${GREEN}passed${NC}"
else
echo -e "testcase $no: ${RED}failed${NC}"
fi
./output ${INPUT_DIR}/input7.txt Out7.txt
val=$(diff Out7.txt ${INPUT_DIR}/output7.txt | wc -l)
if [ $val -eq 0 ]
then
count=$(echo "${count} + 1" | bc -l)
echo -e "testcase $no: ${GREEN}passed${NC}"
else
echo -e "testcase $no: ${RED}failed${NC}"
fi
./output ${INPUT_DIR}/input8.txt Out8.txt
val=$(diff Out8.txt ${INPUT_DIR}/output8.txt | wc -l)
if [ $val -eq 0 ]
then
count=$(echo "${count} + 1" | bc -l)
echo -e "testcase $no: ${GREEN}passed${NC}"
else
echo -e "testcase $no: ${RED}failed${NC}"
fi
./output ${INPUT_DIR}/input9.txt Out9.txt
val=$(diff Out9.txt ${INPUT_DIR}/output9.txt | wc -l)
if [ $val -eq 0 ]
then
count=$(echo "${count} + 1" | bc -l)
echo -e "testcase $no: ${GREEN}passed${NC}"
else
echo -e "testcase $no: ${RED}failed${NC}"
fi
./output ${INPUT_DIR}/input10.txt Out10.txt
val=$(diff Out10.txt ${INPUT_DIR}/output10.txt | wc -l)
if [ $val -eq 0 ]
then
count=$(echo "${count} + 1" | bc -l)
echo -e "testcase $no: ${GREEN}passed${NC}"
else
echo -e "testcase $no: ${RED}failed${NC}"
fi
./output ${INPUT_DIR}/input11.txt Out11.txt
val=$(diff Out11.txt ${INPUT_DIR}/output11.txt | wc -l)
if [ $val -eq 0 ]
then
count=$(echo "${count} + 1" | bc -l)
echo -e "testcase $no: ${GREEN}passed${NC}"
else
echo -e "testcase $no: ${RED}failed${NC}"
fi
./output ${INPUT_DIR}/input12.txt Out12.txt
val=$(diff Out12.txt ${INPUT_DIR}/output12.txt | wc -l)
if [ $val -eq 0 ]
then
count=$(echo "${count} + 1" | bc -l)
echo -e "testcase $no: ${GREEN}passed${NC}"
else
echo -e "testcase $no: ${RED}failed${NC}"
fi
./output ${INPUT_DIR}/input13.txt Out13.txt
val=$(diff Out13.txt ${INPUT_DIR}/output13.txt | wc -l)
if [ $val -eq 0 ]
then
count=$(echo "${count} + 1" | bc -l)
echo -e "testcase $no: ${GREEN}passed${NC}"
else
echo -e "testcase $no: ${RED}failed${NC}"
fi
./output ${INPUT_DIR}/input14.txt Out14.txt
val=$(diff Out14.txt ${INPUT_DIR}/output14.txt | wc -l)
if [ $val -eq 0 ]
then
count=$(echo "${count} + 1" | bc -l)
echo -e "testcase $no: ${GREEN}passed${NC}"
else
echo -e "testcase $no: ${RED}failed${NC}"
fi
marks=$(echo "scale=2; $count / 2.0" | bc -l)
echo "Marks: $marks / 7"
make clean




