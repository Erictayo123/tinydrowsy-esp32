SECTIONS
{
    .embedded_model :
    {
        . = ALIGN(16);
        _binary_tinydrowsy_int8_espdl_start = .;
        *(.tinydrowsy_int8_espdl)
        _binary_tinydrowsy_int8_espdl_end = .;
    } > FLASH
}