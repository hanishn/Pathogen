// Pathogen.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <functional>

#include "portaudio.h"
#include "portmidi.h"
#include "porttime.h"

#ifndef M_PI
#define M_PI  (3.14159265)
#endif

#define M_2PI (M_PI * 2.0)
#define M_PIo2 (M_PI / 2.0)
#define M_iPI (1.0 / M_PI)
#define M_i2PI (1.0 / M_2PI)

typedef unsigned char     UINT_8;
typedef unsigned short    UINT_16;
typedef unsigned int      UINT_32;
typedef uint64_t          UINT_64;

typedef signed char       INT_8;
typedef short             INT_16;
typedef int               INT_32;
typedef int64_t           INT_64;

// TODO: Template on this
typedef float SAMPLE_TYPE;
typedef float TABLE_PHASE_TYPE;

namespace SoundConstants
{
	const UINT_32 c_playbackDuration(60); // seconds
	const UINT_32 c_sampleRate(384000);  // 384KHz
	const UINT_32 c_waveTableSize(8192); // 8K
	const UINT_32 c_maxFundamentalFreq(4000); // 4KHz
	const UINT_32 c_waveTableSizeWithWraparound(c_waveTableSize + 2); //  extra samples for wraparound for interpolation
	const UINT_32 c_FramesPerBuffer(4096);
	const double c_twelveRootOf2(pow(2.0, 1.0 / 12.0));
	const double c_frequency(440.0);
	const double c_initCutoffFrequency(4000.0);
	const double c_maxCutoffFrequency(20000.0);
	const float c_fSampleDuration(1.0f / c_sampleRate); // seconds

	enum AudioChannelEnum : int
	{
		Invalid = -1,
		ChannelBegin = 0,
		Left = ChannelBegin,
		Right,
		ChannelCount
	};

}; // namespace SoundConstants

namespace
{
	const UINT_16 c_messageSize = 256;
};

using namespace SoundConstants;

namespace WaveTables
{
	struct WaveTable
	{
		// TODO: Template on type, size
		SAMPLE_TYPE samples[c_waveTableSizeWithWraparound];
	};

	enum WaveTableType
	{
		Invalid = -1,
		WaveTableTypeBegin = 0,
		Silence = WaveTableTypeBegin,
		Sine,
		Cosine,
//		Square, // 50% PWM
//		Sawtooth,
//		Triangle,
		BandLimitedSquare,
		BandLimitedSawtooth,
		BandLimitedTriangle,
		WaveTableTypeCount
	};

	const char* waveTableNames[WaveTableType::WaveTableTypeCount] =
	{
		"Silence",
		"Sine",
		"Cosine",
//		"Square", // 50% PWM
//		"Sawtooth",
//		"Triangle",
		"BandLimitedSquare",
		"BandLimitedSawtooth",
		"BandLimitedTriangle",
	};

	void GenSilenceWaveTable(WaveTable& waveTable)
	{
		/* initialise sinusoidal wavetable */
		for (INT_32 i = 0; i < c_waveTableSizeWithWraparound; i++)
		{
			waveTable.samples[i] = static_cast<SAMPLE_TYPE>(0);
		}
	}

	void GenSineWaveTable(WaveTable& waveTable)
	{
		/* initialise sinusoidal wavetable */
		double phaseIncr = M_2PI / static_cast<double>(c_waveTableSize);
		double phase = 0.0;
		for (INT_32 i = 0; i < c_waveTableSizeWithWraparound; i++)
		{
			waveTable.samples[i] = static_cast<SAMPLE_TYPE>(sin(phase));
			phase += phaseIncr;
		}
	}

	void GenCosineWaveTable(WaveTable& waveTable)
	{
		/* initialise cosine wavetable */
		double phaseIncr = M_2PI / static_cast<double>(c_waveTableSize);
		double phase = M_PIo2;
		for (INT_32 i = 0; i < c_waveTableSizeWithWraparound; i++)
		{
			waveTable.samples[i] = static_cast<SAMPLE_TYPE>(sin(phase));
			phase += phaseIncr;
		}
	}

	void GenSquareWaveTable(WaveTable& waveTable)
	{
		/* initialise square wavetable */
		double phaseIncr = M_2PI / static_cast<double>(c_waveTableSize);
		double phase = 0.0;
		for (INT_32 i = 0; i < c_waveTableSizeWithWraparound; i++)
		{
			waveTable.samples[i] = static_cast<SAMPLE_TYPE>(
				((phase < M_PI) || (phase > M_2PI)) ? 
 				 1.0 : 
				-1.0);
			phase += phaseIncr;
		}
	}

	void GenSawtoothWaveTable(WaveTable& waveTable)
	{
		/* initialise sawtooth wavetable */
		double phaseIncr = M_2PI / static_cast<double>(c_waveTableSize);
		double phase = 0.0;
		for (INT_32 i = 0; i < c_waveTableSizeWithWraparound; i++)
		{
			SAMPLE_TYPE halfAmpSawtooth = static_cast<SAMPLE_TYPE>(
				((phase < M_PI) || (phase > M_2PI)) ?
				phase * M_iPI :
				1.0 - (phase - M_PI) * M_iPI);
			waveTable.samples[i] = static_cast<SAMPLE_TYPE>(2.0 * halfAmpSawtooth - 1.0);
			phase += phaseIncr;
		}
	}

	void GenTriangleWaveTable(WaveTable& waveTable)
	{
		/* initialise triangle wavetable */
		double phaseIncr = M_2PI / static_cast<double>(c_waveTableSize);
		double phase = 0.0;
		for (INT_32 i = 0; i < c_waveTableSizeWithWraparound; i++)
		{
			waveTable.samples[i] = static_cast<SAMPLE_TYPE>(fmod(phase, M_2PI));
			phase += phaseIncr;
		}
	}

	void GenBandLimitedSquareWaveTable(WaveTable& waveTable)
	{
		/* initialise band limited square wavetable */
		// Square wave is odd harmonics decreasing by inverse of harmonic number

		// c_maxFundamentalFreq
		INT_32 maxHarmonic = static_cast<INT_32>((c_sampleRate / c_maxFundamentalFreq) / 2.0);

		double phaseIncr = M_2PI / static_cast<double>(c_waveTableSize);
		double phase = 0.0;
		for (INT_32 i = 0; i < c_waveTableSizeWithWraparound; i++)
		{
			SAMPLE_TYPE sumOfHarmonics = static_cast<SAMPLE_TYPE>(0.0);
			for (INT_32 harmonic = 1; harmonic <= maxHarmonic; harmonic += 2)
			{
				sumOfHarmonics += static_cast<SAMPLE_TYPE>(sin(phase * harmonic) / harmonic);
			}
			waveTable.samples[i] = sumOfHarmonics;
			phase += phaseIncr;
		}
	}

	void GenBandLimitedSawtoothWaveTable(WaveTable& waveTable)
	{
		/* initialise band limited square wavetable */
		// Sawtooth wave is all harmonics decreasing by inverse of harmonic number

		// c_maxFundamentalFreq
		INT_32 maxHarmonic = static_cast<INT_32>((c_sampleRate / c_maxFundamentalFreq) / 2.0);

		double phaseIncr = M_2PI / static_cast<double>(c_waveTableSize);
		double phase = 0.0;
		for (INT_32 i = 0; i < c_waveTableSizeWithWraparound; i++)
		{
			SAMPLE_TYPE sumOfHarmonics = static_cast<SAMPLE_TYPE>(0.0);
			for (INT_32 harmonic = 1; harmonic <= maxHarmonic; harmonic++)
			{
				sumOfHarmonics += static_cast<SAMPLE_TYPE>(sin(phase * harmonic) / harmonic);
			}
			waveTable.samples[i] = sumOfHarmonics;
			phase += phaseIncr;
		}
	}

	void GenBandLimitedTriangleWaveTable(WaveTable& waveTable)
	{
		/* initialise band limited square wavetable */
		// Triangle wave is odd harmonics decreasing by inverse of harmonic number squared

		// c_maxFundamentalFreq
		INT_32 maxHarmonic = static_cast<INT_32>((c_sampleRate / c_maxFundamentalFreq) / 2.0);

		double phaseIncr = M_2PI / static_cast<double>(c_waveTableSize);
		double phase = 0.0;
		for (INT_32 i = 0; i < c_waveTableSizeWithWraparound; i++)
		{
			SAMPLE_TYPE sumOfHarmonics = static_cast<SAMPLE_TYPE>(0.0);
			for (INT_32 harmonic = 1; harmonic <= maxHarmonic; harmonic += 2)
			{
				sumOfHarmonics += static_cast<SAMPLE_TYPE>(cos(phase * harmonic) / (harmonic * harmonic));
			}
			waveTable.samples[i] = sumOfHarmonics;
			phase += phaseIncr;
		}
	}

	typedef std::function<void(WaveTable&)> WaveTableGenerator;
	WaveTableGenerator waveTableGenerators[WaveTableType::WaveTableTypeCount] = 
	{
		GenSilenceWaveTable,
		GenSineWaveTable,
		GenCosineWaveTable,
//		GenSquareWaveTable,
//		GenSawtoothWaveTable,
//		GenTriangleWaveTable,
		GenBandLimitedSquareWaveTable,
		GenBandLimitedSawtoothWaveTable,
		GenBandLimitedTriangleWaveTable,
	};

	void GenWavetable(WaveTable& waveTable, WaveTableType waveTableType)
	{
		_ASSERT(waveTableType >= 0 && waveTableType < WaveTableType::WaveTableTypeCount);
		waveTableGenerators[waveTableType](waveTable);
	}

	WaveTable waveTables[WaveTableType::WaveTableTypeCount];

	void GenWaveTables()
	{
		for (WaveTableType waveTableType = WaveTableType::WaveTableTypeBegin; waveTableType != WaveTableType::WaveTableTypeCount; waveTableType = WaveTableType((int)waveTableType + 1))
		{
			GenWavetable(waveTables[waveTableType], waveTableType);
		}
	}
}; // namespace WaveTables

using namespace WaveTables;

namespace Tone
{
	const UINT_16 c_RingBufferSize  = 1024;
	const UINT_16 c_maxPolyphony    = 16;
	const UINT_16 c_maxDetuneCents  = 8;
	const UINT_16 c_maxOctaveSwing  = 4;
	const UINT_16 c_OscillatorCount = 9;
	const float c_maxAmplitude      = 0.5f;
	const bool b_useFilter          = true;

	struct Oscillator
	{
		WaveTableType waveTableType = WaveTableType::Sine;

		float amplitude = 0;      // TODO: Maybe this should be SAMPLE_TYPE
		float detuneAmt[c_maxPolyphony];
		INT_8 relativeOctave = 0;   // -3..+3

		// TODO: Implement these
		bool bResync = false;     // Only meaningful for osc2+
	};

	struct Timbre
	{
		Oscillator oscillators[c_OscillatorCount];
		// TODO: ADSR
	};
}; // namespace Tone

using namespace Tone;


class PulseAudioDriver
{
public:
	PulseAudioDriver();

	bool open(PaDeviceIndex index);
	bool close();
	bool start();
	bool stop();

private:

	void ControlRateUpdate(unsigned long framesPerBuffer);

	/* The instance callback, where we have access to every method/variable in object of class Sine */
	int paCallbackMethod(const void *inputBuffer, void *outputBuffer,
		unsigned long framesPerBuffer,
		const PaStreamCallbackTimeInfo* timeInfo,
		PaStreamCallbackFlags statusFlags);

	static int paCallback(const void *inputBuffer, void *outputBuffer,
		unsigned long framesPerBuffer,
		const PaStreamCallbackTimeInfo* timeInfo,
		PaStreamCallbackFlags statusFlags,
		void *userData);

	void paStreamFinishedMethod();

	static void paStreamFinished(void* userData);

	PaStream *stream;
	TABLE_PHASE_TYPE waveTablePhases[AudioChannelEnum::ChannelCount][c_OscillatorCount][c_maxPolyphony] = {};
	UINT_64 frameCount = 0;
	Timbre currentTimbre;
	double currentFrequency = c_frequency;
	double cutoffFrequency  = c_initCutoffFrequency;
	char message[c_messageSize];

	struct RingBuffer
	{
		INT_32 writeHead = 2;
		SAMPLE_TYPE samples[c_RingBufferSize];
	};

	RingBuffer inputRingBuffer;
	RingBuffer outputRingBuffer;

}; // class PulseAudioDriver

PulseAudioDriver::PulseAudioDriver() : stream(0)
{
	GenWaveTables();

	for (UINT_32 i = 0; i < c_RingBufferSize; ++i)
	{
		inputRingBuffer.samples[i]  = 0.0;
		outputRingBuffer.samples[i] = 0.0;
	}

	UINT_32 totalWaveTablePhaseSize = AudioChannelEnum::ChannelCount * c_OscillatorCount * c_maxPolyphony;
	for (UINT_32 index = 0; index < totalWaveTablePhaseSize; ++index)
	{
		((float*)&waveTablePhases[0][0][0])[index] = 0.0f;
	}

	sprintf(message, "No Message");
}

bool PulseAudioDriver::open(PaDeviceIndex index)
{
	PaStreamParameters outputParameters;

	outputParameters.device = index;
	if (outputParameters.device == paNoDevice) {
		return false;
	}

	const PaDeviceInfo* pInfo = Pa_GetDeviceInfo(index);
	if (pInfo != 0)
	{
		printf("Output device name: '%s'\r", pInfo->name);
	}

	outputParameters.channelCount = AudioChannelEnum::ChannelCount;       /* stereo output */
	outputParameters.sampleFormat = paFloat32; /* 32 bit floating point output */ // TODO: Choice of type
	outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
	outputParameters.hostApiSpecificStreamInfo = NULL;

	PaError err = Pa_OpenStream(
		&stream,
		NULL, /* no input */
		&outputParameters,
		c_sampleRate,
		paFramesPerBufferUnspecified,
		paClipOff,      /* we won't output out of range samples so don't bother clipping them */
		&PulseAudioDriver::paCallback,
		this            /* Using 'this' for userData so we can cast to Sine* in paCallback method */
	);

	if (err != paNoError)
	{
		/* Failed to open stream to device !!! */
		return false;
	}

	err = Pa_SetStreamFinishedCallback(stream, &PulseAudioDriver::paStreamFinished);

	if (err != paNoError)
	{
		Pa_CloseStream(stream);
		stream = 0;

		return false;
	}

	return true;
}

bool PulseAudioDriver::close()
{
	if (stream == 0)
		return false;

	PaError err = Pa_CloseStream(stream);
	stream = 0;

	return (err == paNoError);
}

bool PulseAudioDriver::start()
{
	if (stream == 0)
		return false;

	PaError err = Pa_StartStream(stream);

	return (err == paNoError);
}

bool PulseAudioDriver::stop()
{
	if (stream == 0)
		return false;

	PaError err = Pa_StopStream(stream);

	return (err == paNoError);
}

void PulseAudioDriver::ControlRateUpdate(unsigned long framesPerBuffer)
{
	const float noteStepFactor = 2.0f;
	const float tableStepFactor = 0.5f;
	INT_32 noteIndexBefore  = static_cast<INT_32>(c_fSampleDuration * frameCount *  noteStepFactor);
	INT_32 tableIndexBefore = static_cast<INT_32>(c_fSampleDuration * frameCount * tableStepFactor);
	frameCount += framesPerBuffer;
	INT_32 noteIndexAfter   = static_cast<INT_32>(c_fSampleDuration * frameCount *  noteStepFactor);
	INT_32 tableIndexAfter  = static_cast<INT_32>(c_fSampleDuration * frameCount * tableStepFactor);
	if (noteIndexBefore != noteIndexAfter)
	{
		const INT_8 c_notesInOctave = 8;
		INT_8 majorKeyOffset[c_notesInOctave] = { 0, 2, 4, 5, 7, 9, 11, 12 };

		INT_8 noteOffsetFromMiddleC = majorKeyOffset[std::rand() % c_notesInOctave];
		currentFrequency = c_frequency * pow(c_twelveRootOf2, noteOffsetFromMiddleC);
	}
	if (tableIndexBefore != tableIndexAfter)
	{
		for (INT_32 osc = 0; osc < c_OscillatorCount; ++osc)
		{
			currentTimbre.oscillators[osc].waveTableType  = static_cast<WaveTableType>(std::rand() % WaveTableTypeCount);
			currentTimbre.oscillators[osc].amplitude      = c_maxAmplitude * (0.01f * (std::rand() % 100));
			currentTimbre.oscillators[osc].relativeOctave = (std::rand() % (2 * c_maxOctaveSwing + 1)) - c_maxOctaveSwing;

			for (INT_32 voice = 0; voice < c_maxPolyphony; voice++)
			{
				currentTimbre.oscillators[osc].detuneAmt[voice] = 0.001f * ((std::rand() % (2 * c_maxDetuneCents)) - c_maxDetuneCents); // TODO: Set some constants
			}
		}
	}
	if ((noteIndexBefore != noteIndexAfter) || (tableIndexBefore != tableIndexAfter))
	{
		double drand = static_cast<double>(std::rand()) / RAND_MAX;
		double factor = (drand * (c_maxCutoffFrequency - c_frequency)) / c_frequency;
		cutoffFrequency = c_frequency * (1.0f + factor);

		std::cout << "Fundamental Frequency: " << currentFrequency;
		std::cout << std::endl;
		std::cout << "Cutoff Frequency: " << cutoffFrequency;
		std::cout << std::endl;
		for (INT_32 osc = 0; osc < c_OscillatorCount; ++osc)
		{
			std::cout << " Waveform " << osc << ": " << waveTableNames[currentTimbre.oscillators[osc].waveTableType] 
				<< " Amplitude: " << 100 * currentTimbre.oscillators[osc].amplitude 
				<< " Octave: " << ((currentTimbre.oscillators[osc].relativeOctave >=0) ? "+" : "") << static_cast<INT_32>(currentTimbre.oscillators[osc].relativeOctave)
				<< std::endl;
		}
		std::cout << std::endl;
	};
}

/* The instance callback, where we have access to every method/variable in object of class Sine */
int PulseAudioDriver::paCallbackMethod(
	const void *inputBuffer, 
	void *outputBuffer,
	unsigned long framesPerBuffer,
	const PaStreamCallbackTimeInfo* timeInfo,
	PaStreamCallbackFlags statusFlags)
{
	// DONE: Non-integral increment, interpolation
	
	// DONE: Arbitrary table size

	// DONE: Arbitrary sample rate

	// DONE: Arbitrary channel count

	// DONE: Select wavetable

	// DONE: Generate additional waveforms (e.g. triangle, sawtooth, square)

	// DONE: Generate band limited waveforms

	// DONE: Octave offset

	// DONE: Polyphony and detune

	// DONE: Get a standard type header (e.g. uint16)

	// DONE: Wave mixing

	// DONE: Apply filtering, e.g. basic lowpass

	// TODO: MIDI Integration

	// TODO: Factor out parts 
	//	Randomization, Tone Generator, Sampler, Filter, etc.

	// TODO: Resync Osc

	// TODO: ADSR Amplitude Envelope

	// TODO: ADSR Cutoff frequency

	// TODO: Tone sequencing
	//       E.g. Modulation - look at previous N notes and pick a new key that contains all of them 
	//       E.g. Repetition - Occassionally 'step back' a random note count and repeat a certain set of notes before random choice again
	//       E.g. Rhythm     - Not always 1/4 notes, have some rests
	//       E.g. Harmonies  - Generate more than one note - put them in a chord together

	// TODO: Sync Time (use PortTime)

	// TODO: typedef and/or template sample type

	// TODO: MIDI2 Integration

	// TODO: Precalculate frames, just playback in the interrupt handler

	// TODO: Additional voice types (e.g. noise, sampled wavetables)

	// TODO: LFO modulation

	// TODO: Additional filter types

	// TODO: Optimize

	// TODO: Multithread

	// TODO: Move to Compute, in a standard way (e.g. OpenCL)

	// TODO: Effects (e.g. Reverb)

	// TODO: UI

	// TODO: Phase conversion (e.g. PWN --> sqaure, sawtooth --> triangle)

	// TODO: Make an FPGA version

	// Control rate update
	ControlRateUpdate(framesPerBuffer);

	// Init filter
	// Butterworth low-pass filter
	// lambda = 1 / tan(PI * f / sample_rate)
	// a0 = 1/ (1 + 2 * lambda + lambda * lambda)
	// a1 = 2 * s0
	// a2 = a0
	// b1 = 2 * a0 * (1 - lambda * lambda)
	// b2 = a0 * (1 - 2 * lambda + lambda * lambda)
	// y(n) = a0 * x(n) + a1 * x(n - 1) + a2 * x(n - 2) - b1 * y(n - 1) - b2 * y(n - 2)
	const double lambda = 1.0 / tan(M_PI * cutoffFrequency / c_sampleRate);
	const double a0 = 1.0 / (1.0 + 2.0 * lambda + lambda * lambda);
	const double a1 = 2.0 * a0;
	const double a2 = a0;
	const double b1 = 2.0 * a0 * (1.0 - lambda * lambda);
	const double b2 = a0 * (1.0 - 2.0 * lambda + lambda * lambda);


	// Sample rate update
	{
		// One pass thru wave table is 1 periodic cycle
		// c_waveTableSize     = samples / cycle
		// c_frequency         =  cycles / second
		// c_sampleRate        =  frames / second
		// c_fSampleDuration   = seconds /  frame
		// We want samples / frame
		// c_waveTableSize  * c_frequency * c_fSampleDuration 
		SAMPLE_TYPE *out = (SAMPLE_TYPE*)outputBuffer;

		float tableIncrement[c_OscillatorCount];
		for (INT_32 osc = 0; osc < c_OscillatorCount; ++osc)
		{
			tableIncrement[osc] = static_cast<float>(c_waveTableSize * currentFrequency * c_fSampleDuration);

			for (INT_32 i = 0; i < currentTimbre.oscillators[osc].relativeOctave; ++i)
			{
				tableIncrement[osc] *= 2.0f;
			}
			for (INT_32 i = 0; i < -currentTimbre.oscillators[osc].relativeOctave; ++i)
			{
				tableIncrement[osc] /= 2.0f;
			}
		}

		WaveTables::WaveTable* l_waveTables[c_OscillatorCount];
		for (UINT_32 osc = 0; osc < c_OscillatorCount; ++osc)
		{
			l_waveTables[osc] = &waveTables[currentTimbre.oscillators[osc].waveTableType];
		}

		for (UINT_32 i = 0; i < framesPerBuffer; i++)
		{
			for(AudioChannelEnum channel = AudioChannelEnum::ChannelBegin; channel != AudioChannelEnum::ChannelCount; channel = AudioChannelEnum((int)channel + 1))
			{
				SAMPLE_TYPE interpolatedSample = 0.0f;

				for (UINT_32 osc = 0; osc < c_OscillatorCount; ++osc)
				{
					for (UINT_32 voice = 0; voice < c_maxPolyphony; voice++)
					{
						UINT_32 phaseFloor = static_cast<UINT_32>(waveTablePhases[channel][osc][voice]);
						float ratio = waveTablePhases[channel][osc][voice] - phaseFloor;

						SAMPLE_TYPE sampleFloor = (*l_waveTables[osc]).samples[phaseFloor];
						SAMPLE_TYPE sampleCeil  = (*l_waveTables[osc]).samples[phaseFloor + 1];
						interpolatedSample += currentTimbre.oscillators[osc].amplitude * (sampleFloor + ratio * (sampleCeil - sampleFloor)) /
							static_cast<float>(c_OscillatorCount * c_maxPolyphony);

						float detunedIncrement = tableIncrement[osc] * (1.0f + currentTimbre.oscillators[osc].detuneAmt[voice]);
						detunedIncrement = (detunedIncrement > 0.0f) ? detunedIncrement : 0.0f;
						detunedIncrement = (detunedIncrement < c_waveTableSize) ? detunedIncrement : c_waveTableSize;
						waveTablePhases[channel][osc][voice] += detunedIncrement;

						// TODO: fmod
						while (waveTablePhases[channel][osc][voice] >= c_waveTableSize)
						{
							waveTablePhases[channel][osc][voice] -= c_waveTableSize;
						}
					}
				}

				// Apply filtering
				inputRingBuffer.samples[inputRingBuffer.writeHead] = interpolatedSample;
				
				// TODO: MOD is heavy. Maybe don't use it.
				INT_32 inputWriteMinus1  = (c_RingBufferSize + inputRingBuffer.writeHead - 1) % c_RingBufferSize;
				INT_32 inputWriteMinus2  = (c_RingBufferSize + inputRingBuffer.writeHead - 2) % c_RingBufferSize;
				INT_32 outputWriteMinus1 = (c_RingBufferSize + outputRingBuffer.writeHead - 1) % c_RingBufferSize;
				INT_32 outputWriteMinus2 = (c_RingBufferSize + outputRingBuffer.writeHead - 2) % c_RingBufferSize;
				 
				SAMPLE_TYPE x0 = static_cast<SAMPLE_TYPE>( a0 *  inputRingBuffer.samples[inputRingBuffer.writeHead]);
				SAMPLE_TYPE x1 = static_cast<SAMPLE_TYPE>( a1 *  inputRingBuffer.samples[inputWriteMinus1]);
				SAMPLE_TYPE x2 = static_cast<SAMPLE_TYPE>( a2 *  inputRingBuffer.samples[inputWriteMinus2]);
				SAMPLE_TYPE y1 = static_cast<SAMPLE_TYPE>(-b1 * outputRingBuffer.samples[outputWriteMinus1]);
				SAMPLE_TYPE y2 = static_cast<SAMPLE_TYPE>(-b2 * outputRingBuffer.samples[outputWriteMinus2]);
				SAMPLE_TYPE filteredOutput = x0 + x1 + x2 + y1 + y2;

				outputRingBuffer.samples[outputRingBuffer.writeHead] = b_useFilter ? filteredOutput : interpolatedSample;

				*(out++) = filteredOutput;

				inputRingBuffer.writeHead  = (inputRingBuffer.writeHead + 1) % c_RingBufferSize;
				outputRingBuffer.writeHead = (outputRingBuffer.writeHead + 1) % c_RingBufferSize;
			}
		}
	}

	return paContinue;
}

/* This routine will be called by the PortAudio engine when audio is needed.
** It may called at interrupt level on some machines so don't do anything
** that could mess up the system like calling malloc() or free().
*/
int PulseAudioDriver::paCallback(
	const void *inputBuffer, 
	void *outputBuffer,
	unsigned long framesPerBuffer,
	const PaStreamCallbackTimeInfo* timeInfo,
	PaStreamCallbackFlags statusFlags,
	void *userData)
{
	/* Here we cast userData to Sine* type so we can call the instance method paCallbackMethod, we can do that since
	we called Pa_OpenStream with 'this' for userData */
	return ((PulseAudioDriver*)userData)->paCallbackMethod(
		inputBuffer, 
		outputBuffer,
		framesPerBuffer,
		timeInfo,
		statusFlags);
}


void PulseAudioDriver::paStreamFinishedMethod()
{
	printf("Stream Completed: %s\n", message);
}

/*
* This routine is called by portaudio when playback is done.
*/
void PulseAudioDriver::paStreamFinished(void* userData)
{
	return ((PulseAudioDriver*)userData)->paStreamFinishedMethod();
}

/*******************************************************************/
class ScopedPulseAudioHandler
{
public:
	ScopedPulseAudioHandler()
		: _result(Pa_Initialize())
	{
	}
	~ScopedPulseAudioHandler()
	{
		if (_result == paNoError)
		{
			Pa_Terminate();
		}
	}

	PaError result() const { return _result; }

private:
	PaError _result;
};

class ScopedPulseMidiHandler
{
public:
	ScopedPulseMidiHandler()
		: _result(Pm_Initialize())
	{
	}
	~ScopedPulseMidiHandler()
	{
		if (_result == paNoError)
		{
			Pm_Terminate();
		}
	}

	PmError result() const { return _result; }

private:
	PmError _result;
};


/*******************************************************************/
int main(void)
{
	PulseAudioDriver pulseAudioDriver;

	printf("PortAudio Test: output wavetable. SR = %d, BufSize = %d\n", c_sampleRate, c_FramesPerBuffer);

	ScopedPulseAudioHandler paInit;
	if (paInit.result() != paNoError)
	{
		fprintf(stderr, "An error occured while using the PulseAudio stream\n");
		fprintf(stderr, "Error number: %d\n", paInit.result());
		fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(paInit.result()));
		return 1;
	}

	ScopedPulseMidiHandler pmInit;
	if (pmInit.result() != pmNoError)
	{
		fprintf(stderr, "An error occured while using the PulseMIDI stream\n");
		fprintf(stderr, "Error number: %d\n", pmInit.result());
		fprintf(stderr, "Error message: %s\n", Pm_GetErrorText(pmInit.result()));
		return 1;
	}

	// Init MIDI Devices
	{
		UINT_16 numMIDIDev = Pm_CountDevices();
		PmDeviceID defaultMIDIDev = Pm_GetDefaultInputDeviceID();
		printf("There are %d MIDI devices connected. Default device ID %d\n",
			numMIDIDev, defaultMIDIDev);

		for (UINT_16 midiIndex = 0; midiIndex < numMIDIDev; ++midiIndex)
		{
			const PmDeviceInfo* midiDevInfo = Pm_GetDeviceInfo(midiIndex);
			if (midiDevInfo)
			{
				printf("MIDI Device Index %u. Name %s. Interface %s. Input %d. Output %d. Opened %d \n",
					midiIndex, midiDevInfo->name, midiDevInfo->interf, midiDevInfo->input, midiDevInfo->output, midiDevInfo->opened);
			}
		}

		if (defaultMIDIDev)
		{
			PortMidiStream *portMidiStream;
			void* midiDevDriverInf = nullptr;
			PmError pmError;
			INT_32 bufferSize = 1024;
			PmTimeProcPtr pmTimeProcPtr = nullptr;
			void* timeInfo = nullptr;
		    Pt_Start(1, nullptr, nullptr); // Need a time function
//			pmError = Pm_OpenInput(&portMidiStream, defaultMIDIDev, midiDevDriverInf, bufferSize, pmTimeProcPtr, timeInfo);
//			if (pmInit.result() != pmNoError)
//			{
//				printf("Error in Pm_OpenInput for device %d.\n", defaultMIDIDev);
//				return 1;
//			}
		}
	}

	if (pulseAudioDriver.open(Pa_GetDefaultOutputDevice()))
	{
		if (pulseAudioDriver.start())
		{
			printf("Play for %d seconds.\n", c_playbackDuration);
			Pa_Sleep(c_playbackDuration * 1000);

			pulseAudioDriver.stop();
		}

		pulseAudioDriver.close();
	}

	printf("Test finished.\n");
	return paNoError;
}
