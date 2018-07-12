echo "Starting Xvfb.."
Xvfb :99 &
echo $! > ./Xvfb.pid
export DISPLAY=:99
echo "Running check.py.."
python3 ./check.py
echo "Stopping Xvfb.."
kill `cat ./Xvfb.pid`
rm ./Xvfb.pid
