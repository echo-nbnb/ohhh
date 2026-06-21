import SpiritPage from "../SpiritPage";
import { demoCharacter } from "./demoCharacter";

export default function Demo() {
  return (
    <SpiritPage
      character={demoCharacter}
      onComplete={(character) => {
        console.log("唤灵完成", character);
      }}
    />
  );
}
