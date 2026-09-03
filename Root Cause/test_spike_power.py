# test_spike_power.py

# Candidate thresholds from our dataset
TEMP_THRESHOLD = 1.1677
HUMIDITY_THRESHOLD = 3.4070
PRESSURE_THRESHOLD = 1.5923


def check_spikes(temp_change, humidity_change, pressure_change):

    temperature_spike = abs(temp_change) > TEMP_THRESHOLD
    humidity_spike = abs(humidity_change) > HUMIDITY_THRESHOLD
    pressure_spike = abs(pressure_change) > PRESSURE_THRESHOLD

    spike_count = (
        temperature_spike
        + humidity_spike
        + pressure_spike
    )

    return {
        "temperature_spike": temperature_spike,
        "humidity_spike": humidity_spike,
        "pressure_spike": pressure_spike,
        "spike_count": spike_count
    }


# Test 1: Normal changes
print("Test 1 - Normal:")
print(check_spikes(0.3, 1.0, 0.5))


# Test 2: Temperature spike
print("\nTest 2 - Temperature spike:")
print(check_spikes(10.0, 0.2, 0.1))


# Test 3: Humidity spike
print("\nTest 3 - Humidity spike:")
print(check_spikes(0.2, 20.0, 0.3))


# Test 4: Pressure spike
print("\nTest 4 - Pressure spike:")
print(check_spikes(0.2, 0.3, 10.0))


# Test 5: All sensors spike
print("\nTest 5 - All sensors spike:")
print(check_spikes(10.0, 20.0, 10.0))