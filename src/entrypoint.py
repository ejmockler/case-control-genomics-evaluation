from input import processInputFiles
from predict import classify
import asyncio
import pickle


async def main():
    (
        caseGenotypes,
        caseIDs,
        controlGenotypes,
        controlIDs,
        clinicalData,
    ) = await processInputFiles()
    results = await classify(
        caseGenotypes, caseIDs, controlGenotypes, controlIDs, clinicalData
    )
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
    pickle.dump(results, open("results.pkl", "wb"))
