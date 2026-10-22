/* Memory layout for STM32WLE5JC (256K Flash, 64K RAM) */
MEMORY
{
  FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 256K
  RAM (rwx)  : ORIGIN = 0x20000000, LENGTH = 64K
}

/* Reserve stack at the end of RAM */
_stack_start = ORIGIN(RAM) + LENGTH(RAM);
